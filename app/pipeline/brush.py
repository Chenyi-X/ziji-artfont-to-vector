"""Compact monochrome brush modulation plus independently editable texture marks."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .svgutil import SvgDocument,SvgLayer,FittedContour,polygon_area,contour_to_path_d,rgb_to_hex,build_linear_gradient
from .curvefit import fit_ring


def brush_candidate(prepared, options):
    if options.glyph_count!=1 or prepared.bg_color is None or not prepared.quality.background_uniform or options.spot_colors:
        return None
    raw=prepared.rgb_raw.astype(float);background=np.asarray(prepared.bg_color)
    distance=np.linalg.norm(raw-background,axis=2)
    selected=raw[distance>100]
    if len(selected)<100:return None
    ink=np.percentile(selected,1,axis=0)
    direction=ink-background;denominator=direction@direction
    if denominator<1000:return None
    alpha=np.clip((raw-background)@direction/denominator,0,1)
    reconstructed=background+alpha[:,:,None]*direction
    if np.abs(reconstructed[distance>20]-raw[distance>20]).mean()>2:return None
    mask=(distance>20).astype(np.uint8)*255
    core=cv2.erode(mask,np.ones((5,5),np.uint8))>0
    if core.sum()<100:return None
    h,w=mask.shape;columns=np.full(w,np.nan)
    for x in range(w):
        values=alpha[:,x][core[:,x]]
        if len(values)>3:columns[x]=np.median(values)
    valid=np.isfinite(columns)
    if valid.sum()<20:return None
    columns=gaussian_filter1d(np.interp(np.arange(w),np.flatnonzero(valid),columns[valid]),1)
    if np.ptp(columns)<.12:return None
    document=SvgDocument(w,h);pieces=[];nodes=0
    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        points=contour.reshape(-1,2).astype(float)+.5
        if len(points)<3:continue
        segments=fit_ring(points,1.4,62)
        nodes+=len(segments)+1
        pieces.append(contour_to_path_d(FittedContour(segments,points,abs(polygon_area(points)))))
    stops=[(x/max(1,w-1),rgb_to_hex(background*(1-columns[x])+ink*columns[x])) for x in range(w)]
    document.defs.append(build_linear_gradient('brush-ink',stops,0,0,w-1,0))
    document.layers.append(SvgLayer('#000000',[' '.join(pieces)],label='brush-body',gradient={'ref':'url(#brush-ink)','stops':stops}))
    marks=0
    for dark in [True,False]:
        local=(((alpha-columns[None,:]>.15) if dark else (alpha-columns[None,:]<-.15))&core).astype(np.uint8)*255
        local=(cv2.GaussianBlur(local,(3,3),0)>100).astype(np.uint8)
        n,labels,stats,_=cv2.connectedComponentsWithStats(local,8)
        color=rgb_to_hex(ink if dark else background*.35+ink*.65)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]<5:continue
            y,x=np.nonzero(labels==i);points=np.c_[x,y].astype(float)+.5;center=points.mean(axis=0)
            _,vectors=np.linalg.eigh(np.cov(points.T));axis=vectors[:,-1];normal=vectors[:,0]
            length=np.ptp((points-center)@axis)+1;width=max(1.,np.ptp((points-center)@normal)+1)
            width=min(width,max(1.,len(points)/max(1,length-width+np.pi*width/4)))
            extent=max(0.,(length-width)/2);a=center-axis*extent;b=center+axis*extent
            d=f'M{a[0]:.2f} {a[1]:.2f} L{b[0]:.2f} {b[1]:.2f}'
            document.layers.append(SvgLayer('none',[d],label='brush-texture',stroke=color,stroke_width=width))
            marks+=1;nodes+=2
    if not 50<=nodes<=options.max_nodes:return None
    return document,{'engine':'contour','layers':len(document.layers),'glyphs':1,'node_total':nodes,
                     'brush_texture_marks':marks,'brush_candidate':True}
