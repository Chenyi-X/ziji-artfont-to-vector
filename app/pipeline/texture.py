"""Local texture agreement for monochrome brush candidates."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def texture_agreement(prepared,pixels):
    raw=prepared.rgb_raw.astype(float);background=np.asarray(prepared.bg_color)
    distance=np.linalg.norm(raw-background,axis=2)
    ink=np.percentile(raw[distance>100],1,axis=0);direction=ink-background
    alpha=np.clip((raw-background)@direction/(direction@direction),0,1)
    mask=(distance>20).astype(np.uint8)*255
    core=cv2.erode(mask,np.ones((5,5),np.uint8))>0
    columns=np.full(mask.shape[1],np.nan)
    for x in range(mask.shape[1]):
        values=alpha[:,x][core[:,x]]
        if len(values)>3:columns[x]=np.median(values)
    valid=np.isfinite(columns)
    if valid.sum()<2:return {'passed':False,'reason':'insufficient_reference'}
    columns=gaussian_filter1d(np.interp(np.arange(len(columns)),np.flatnonzero(valid),columns[valid]),1)
    coverage=pixels[:,:,3:4].astype(float)/255
    painted=pixels[:,:,:3]*coverage+background*(1-coverage)
    result_alpha=np.clip((painted-background)@direction/(direction@direction),0,1)
    records={}
    for name,sign in [('dark',1),('light',-1)]:
        reference=(sign*(alpha-columns[None,:])>.15)&core
        result=(sign*(result_alpha-columns[None,:])>.15)&core
        overlap=np.logical_and(reference,result).sum()
        precision=float(overlap/max(1,result.sum()))
        recall=float(overlap/max(1,reference.sum()))
        f1=2*precision*recall/max(1e-9,precision+recall)
        records[name]={'precision':precision,'recall':recall,'f1':f1,
                       'source_pixels':int(reference.sum()),'rendered_pixels':int(result.sum())}
    return {'passed':all(item['f1']>=.85 for item in records.values()),'regions':records,
            'scope':'core_texture_pixels_relative_to_column_median','threshold':.85}
