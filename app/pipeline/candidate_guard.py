"""Accept geometry candidates only after independent rendering and local comparisons."""
from __future__ import annotations
import cv2
import numpy as np


def independent_raster(document):
    import fitz
    from .export import svg_to_pdf
    with fitz.open(stream=svg_to_pdf(document.to_svg()),filetype='pdf') as pdf:
        pix=pdf[0].get_pixmap(alpha=True)
        pixels=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,4).copy()
    alpha=pixels[:,:,3:4].astype(float)
    pixels[:,:,:3]=np.clip(np.divide(pixels[:,:,:3].astype(float)*255,alpha,
                           out=np.zeros_like(pixels[:,:,:3],dtype=float),where=alpha>0),0,255).astype(np.uint8)
    return pixels


def compare_candidate(prepared,options,document,stats,candidate,candidate_stats):
    from .topology import compare_topology
    if any(layer.filter_id for layer in document.layers+candidate.layers):
        return {'rejected':['filter_effect'],'renderer':None}
    raw=prepared.rgb_raw.astype(float);background=np.asarray(prepared.bg_color)
    support=np.linalg.norm(raw-background,axis=2)>20
    a=independent_raster(document);b=independent_raster(candidate)
    region=cv2.dilate(support.astype(np.uint8),np.ones((3,3),np.uint8))>0
    core=cv2.erode((prepared.mask>127).astype(np.uint8),np.ones((3,3),np.uint8),iterations=3)>0
    def measures(pixels):
        alpha=pixels[:,:,3:4].astype(float)/255
        rendered=pixels[:,:,:3]*alpha+background*(1-alpha)
        errors=np.abs(rendered-raw).mean(axis=2)/255
        mask=pixels[:,:,3]>127
        iou=np.logical_and(support,mask).sum()/max(1,np.logical_or(support,mask).sum())
        return float(iou),float(errors[region].mean()),float(errors[core].mean()) if core.any() else 0
    ma,mb=measures(a),measures(b);reasons=[]
    if mb[0]<ma[0]:reasons.append('source_shape')
    if mb[1]>ma[1] or mb[2]>ma[2]:reasons.append('source_color')
    cap=options.max_nodes*max(1,options.glyph_count or stats.get('glyphs',1))
    if candidate_stats.get('node_total',0)>max(cap,stats.get('node_total',0)):reasons.append('nodes')
    if stats.get('shadow_layers',0) and not candidate_stats.get('shadow_layers',0):reasons.append('shadow_layers')
    if not reasons:
        before=compare_topology(support,a[:,:,3]>127);after=compare_topology(support,b[:,:,3]>127)
        for kind in ('components','holes'):
            for old,new in zip(before[kind]['regions'],after[kind]['regions']):
                if old['status']=='pass' and new['status']!='pass':reasons.append('local_'+kind)
            if len(after[kind]['extra_regions'])>len(before[kind]['extra_regions']):reasons.append('extra_'+kind)
    return {'renderer':'native_pdf_mupdf','before':ma,'candidate':mb,'rejected':reasons}
