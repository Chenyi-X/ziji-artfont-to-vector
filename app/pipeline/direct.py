"""Source-pixel silhouettes with bounded cubic approximation for clean lettering."""
from __future__ import annotations
import cv2
import numpy as np
from .curvefit import fit_ring
from .svgutil import SvgDocument, SvgLayer, FittedContour, contour_to_path_d, polygon_area, rgb_to_hex


def fit_mask_paths(mask, tolerance=.6):
    """Fit pixel-cell boundaries and keep each outer ring with its holes."""
    up = cv2.resize(mask.astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    contours, hierarchy = cv2.findContours(up, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], 0
    pieces = {}
    nodes = 0
    for index, contour in enumerate(contours):
        points = (contour.reshape(-1, 2).astype(float) + .5) / 4
        if len(points) < 3:
            continue
        segments = fit_ring(points, tolerance, 62)
        if not segments:
            continue
        nodes += len(segments) + 1
        pieces[index] = contour_to_path_d(FittedContour(segments, points, abs(polygon_area(points))))
    paths = []
    for index, entry in enumerate(hierarchy[0]):
        depth, parent = 0, int(entry[3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0][parent][3])
        if depth % 2 or index not in pieces:
            continue
        rings = [pieces[index]]
        child = int(entry[2])
        while child >= 0:
            if child in pieces:
                rings.append(pieces[child])
            child = int(hierarchy[0][child][0])
        paths.append(' '.join(rings))
    return paths, nodes


def fit_mask(mask, tolerance=.6):
    paths, nodes = fit_mask_paths(mask, tolerance)
    return ' '.join(paths), nodes


def monochrome_document(prepared, options):
    if (not prepared.quality.background_uniform or prepared.quality.textured_background
            or options.spot_colors or prepared.bg_color is None):
        return None
    raw = prepared.rgb_raw.astype(float)
    bg = np.asarray(prepared.bg_color)
    transparent = prepared.source_rgba is not None and np.any(prepared.source_rgba[:, :, 3] < 250)
    if transparent:
        support = prepared.source_rgba[:, :, 3] > 127
    else:
        support = np.linalg.norm(raw - bg, axis=2) > 20
    core = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if core.sum() < 100:
        return None
    colors = raw[core]
    ink = np.median(colors, axis=0)
    direction = ink-bg
    denom = float(direction@direction)
    if denom < 1000:
        return None
    alpha = np.clip((raw-bg)@direction/denom, 0, 1)
    if np.abs(raw[core]-(bg+alpha[:, :, None]*direction)[core]).mean() > 2:
        return None
    # Broad tonal modulation needs a separate texture representation.
    if np.percentile(np.linalg.norm(colors-ink, axis=1), 95) > 10:
        return None
    glyphs = options.glyph_count or 1
    from .native_render import render_svg
    from .topology import compare_topology
    for tol in [.75, .55, .35]:
        paths, nodes = fit_mask_paths(support, tol)
        if not paths or nodes > options.max_nodes*glyphs or (options.glyph_count and nodes < 50*glyphs):
            continue
        doc = SvgDocument(support.shape[1], support.shape[0])
        doc.layers.append(SvgLayer(rgb_to_hex(ink), paths, label='lettering'))
        pixels = render_svg(doc.to_svg())
        output = pixels[:, :, 3] > 127
        iou = np.logical_and(support, output).sum()/max(1, np.logical_or(support, output).sum())
        topology = compare_topology(support, output)
        if iou >= .97 and topology['passed']:
            return doc, {'engine':'contour', 'source_silhouette':True, 'node_total':nodes,
                         'glyphs':glyphs, 'layers':1, 'source_iou':round(float(iou),4),
                         'source_topology_passed':True}
    return None
