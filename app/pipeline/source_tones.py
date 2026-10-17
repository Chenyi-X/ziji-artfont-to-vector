"""Editable monochrome edge tones selected against source pixels and local topology."""
from __future__ import annotations

import cv2
import numpy as np

from .candidate_guard import compare_candidate, independent_raster
from .curvefit import fit_ring
from .svgutil import (
    FittedContour, SvgDocument, SvgLayer, contour_to_path_d, polygon_area,
    rgb_to_hex,
)
from .topology import compare_topology


def select_source_tones(prepared, options, document, stats):
    """Try nested edge tones only for opaque, unrotated monochrome material."""
    if (prepared.bg_color is None or prepared.rgb_raw is None
            or prepared.rot_matrix is not None
            or not prepared.quality.background_uniform
            or prepared.quality.textured_background
            or prepared.quality.is_colorful or options.spot_colors
            or any(layer.filter_id or layer.gradient for layer in document.layers)):
        return document, stats
    if (prepared.source_rgba is None
            or np.any(prepared.source_rgba[:, :, 3] < 255)):
        return document, stats

    raw = prepared.rgb_raw.astype(float)
    background = np.asarray(prepared.bg_color)
    distance = np.linalg.norm(raw - background, axis=2)
    selected = raw[distance > 100]
    if len(selected) < 100:
        return document, stats
    ink = np.percentile(selected, 1, axis=0)
    direction = ink - background
    denominator = float(direction @ direction)
    if denominator < 1000:
        return document, stats
    alpha = np.clip((raw - background) @ direction / denominator, 0, 1)
    support = distance > 20
    reconstructed = background + alpha[:, :, None] * direction
    if np.abs(reconstructed[support] - raw[support]).mean() > 2:
        return document, stats
    # Broad modulation belongs to a gradient or texture model.
    solid = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if not solid.any() or np.percentile(alpha[solid], 10) < .95:
        return document, stats

    glyphs = max(1, options.glyph_count or stats.get('glyphs', 1))
    cap = options.max_nodes * glyphs
    h, w = support.shape
    candidate = SvgDocument(w, h)
    candidate.metadata = dict(document.metadata)
    boundaries = [20 / np.sqrt(denominator), .45, .85]
    nodes = 0
    for index, boundary in enumerate(boundaries):
        mask = (alpha > boundary).astype(np.uint8) * 255
        next_boundary = boundaries[index + 1] if index + 1 < len(boundaries) else 1.1
        ring = (mask > 0) & (alpha <= next_boundary)
        color = raw[ring].mean(axis=0) if ring.any() else ink
        expanded = cv2.resize(mask, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(expanded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        paths = []
        for contour in contours:
            points = (contour.reshape(-1, 2).astype(float) + .5) / 4
            if len(points) < 3:
                continue
            segments = fit_ring(points, .8 if index == 0 else 1.1, 62)
            nodes += len(segments) + 1
            if nodes > cap:
                return document, {**stats, 'source_tone_candidate': {
                    'rejected': ['nodes'], 'node_total_lower_bound': nodes}}
            paths.append(contour_to_path_d(FittedContour(
                segments, points, abs(polygon_area(points)))))
        if paths:
            candidate.layers.append(SvgLayer(
                rgb_to_hex(color), [' '.join(paths)], label='source-edge-tone'))
    if not candidate.layers or nodes < options.min_nodes * glyphs:
        return document, stats

    candidate_stats = {
        'engine': 'contour', 'glyphs': glyphs, 'layers': len(candidate.layers),
        'node_total': nodes, 'shadow_layers': 0,
    }
    diagnostic = compare_candidate(
        prepared, options, document, stats, candidate, candidate_stats)
    if not diagnostic['rejected']:
        raster = independent_raster(candidate)
        topology = compare_topology(support, raster[:, :, 3] > 127)
        diagnostic['source_topology'] = topology
        if not topology['passed']:
            diagnostic['rejected'].append('source_topology')
    if diagnostic['rejected']:
        return document, {**stats, 'source_tone_candidate': diagnostic}
    candidate_stats.update({
        'source_tones_selected': True, 'source_tone_candidate': diagnostic,
        'previous_node_total': stats.get('node_total'),
    })
    return candidate, candidate_stats
