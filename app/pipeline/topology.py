"""Match foreground components and enclosed holes against an aligned mask."""
from __future__ import annotations

import cv2
import numpy as np


def _regions(mask: np.ndarray, *, holes: bool, min_area: int):
    binary = (~mask if holes else mask).astype(np.uint8)
    # Dual connectivity keeps diagonal foreground contacts from opening a hole.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=4 if holes else 8)
    excluded = set()
    if holes:
        excluded.update(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])).tolist())
    keep = [i for i in range(1, count)
            if i not in excluded and stats[i, cv2.CC_STAT_AREA] >= min_area]
    lookup = np.zeros(count, dtype=np.int32)
    regions = []
    for new_id, old_id in enumerate(keep, 1):
        lookup[old_id] = new_id
        regions.append({'id': new_id, 'area': int(stats[old_id, cv2.CC_STAT_AREA]),
                        'bbox': [int(v) for v in stats[old_id, :4]]})
    return lookup[labels], regions


def _match(reference: np.ndarray, rendered: np.ndarray, *, holes: bool,
           min_area: int, min_iou: float) -> dict:
    src_labels, src = _regions(reference, holes=holes, min_area=min_area)
    dst_labels, dst = _regions(rendered, holes=holes, min_area=min_area)
    both = (src_labels > 0) & (dst_labels > 0)
    radix = len(dst) + 1
    codes, intersections = np.unique(
        src_labels[both].astype(np.int64) * radix + dst_labels[both], return_counts=True)
    by_src = [[] for _ in src]
    by_dst = [[] for _ in dst]
    for code, intersection in zip(codes.tolist(), intersections.tolist()):
        si, di = divmod(code, radix)
        by_src[si - 1].append((di, intersection))
        by_dst[di - 1].append(si)
    rows = []
    for region, overlaps in zip(src, by_src):
        row = dict(region, matched_ids=[pair[0] for pair in overlaps],
                   intersection=0, recall=0.0, precision=0.0, iou=0.0)
        if not overlaps:
            row['status'] = 'missing'
        elif len(overlaps) > 1:
            row['status'] = 'split'
        else:
            di, intersection = overlaps[0]
            target = dst[di - 1]
            iou = intersection / (region['area'] + target['area'] - intersection)
            row.update(intersection=intersection, recall=intersection / region['area'],
                       precision=intersection / target['area'], iou=iou)
            row['status'] = ('merged' if len(by_dst[di - 1]) > 1 else
                             'pass' if iou >= min_iou else 'deformed')
        rows.append(row)
    extra = [region for region, overlaps in zip(dst, by_dst) if not overlaps]
    return {'passed': all(row['status'] == 'pass' for row in rows) and not extra,
            'reference_count': len(src), 'rendered_count': len(dst),
            'regions': rows, 'extra_regions': extra}


def compare_topology(reference: np.ndarray, rendered: np.ndarray, *,
                     min_area: int = 4, min_iou: float = 0.85) -> dict:
    """Diagnose region changes using fixed coordinates and one-to-one overlaps.

    The 4-pixel region floor and 85% per-region IoU are diagnostic settings.
    Features below the floor and visual stroke-width quality need separate review.
    """
    reference = np.asarray(reference)
    rendered = np.asarray(rendered)
    if reference.ndim != 2 or reference.shape != rendered.shape or not reference.size:
        raise ValueError('Topology masks must be nonempty 2D arrays of the same shape')
    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 1:
        raise ValueError('min_area must be a positive integer')
    if isinstance(min_iou, bool) or not np.isfinite(min_iou) or not 0 < min_iou <= 1:
        raise ValueError('min_iou must be finite and in (0, 1]')
    reference, rendered = reference.astype(bool), rendered.astype(bool)
    components = _match(reference, rendered, holes=False, min_area=min_area, min_iou=min_iou)
    holes = _match(reference, rendered, holes=True, min_area=min_area, min_iou=min_iou)
    return {'scope': 'aligned_binary_region_diagnostic',
            'passed': components['passed'] and holes['passed'],
            'settings': {'min_region_pixels': min_area, 'min_region_iou': min_iou,
                         'foreground_connectivity': 8, 'background_connectivity': 4},
            'components': components, 'holes': holes}
