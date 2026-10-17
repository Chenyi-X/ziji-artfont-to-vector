"""Propose source-supported edges and retain only constrained improvements."""
from __future__ import annotations
from dataclasses import replace
import numpy as np


def select_source_edges(prepared, options, document, stats):
    from .vectorize import vectorize_contour
    from .candidate_guard import compare_candidate
    if prepared.bg_color is None or not prepared.quality.background_uniform or prepared.quality.textured_background:
        return document,stats
    if options.spot_colors or any(layer.filter_id for layer in document.layers):
        return document,stats
    support=np.linalg.norm(prepared.rgb_raw.astype(float)-np.asarray(prepared.bg_color),axis=2)>20
    if support.sum()<100:return document,stats
    mask=prepared.mask>127
    overlap=np.logical_and(mask,support).sum()/max(1,np.logical_or(mask,support).sum())
    if overlap>.995:return document,stats
    alternative=replace(prepared,mask=support.astype(np.uint8)*255,
                        rgba=np.dstack([prepared.rgb_raw[:,:,::-1],prepared.soft_alpha]))
    candidate,candidate_stats=vectorize_contour(alternative,replace(options,contour_coordinate_offset=.25,preserve_source_gaps=True))
    if not candidate.layers:return document,stats
    diagnostic=compare_candidate(prepared,options,document,stats,candidate,candidate_stats)
    diagnostic['source_mask_overlap']=float(overlap)
    if diagnostic['rejected']:
        stats['source_edge_candidate']=diagnostic
        return document,stats
    candidate_stats['source_edge_candidate']=diagnostic
    candidate_stats['source_edges_selected']=True
    return candidate,candidate_stats
