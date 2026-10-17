"""Least-squares cubic fitting with a bounded sample error and corner breaks."""
from __future__ import annotations
import numpy as np


def _unit(v):
    length = float(np.linalg.norm(v))
    return v / length if length > 1e-12 else np.array([1.0,0.0])


def _fit_span(points, tolerance, depth=0):
    p = np.asarray(points,dtype=float)
    if len(p)<2:
        return []
    chord = p[-1]-p[0]
    length = float(np.linalg.norm(chord))
    if len(p)==2:
        return [(p[0],None,None,p[-1])]
    distances = np.linalg.norm(np.diff(p,axis=0),axis=1)
    u = np.r_[0.0,np.cumsum(distances)]
    if u[-1]<1e-10:
        return []
    u /= u[-1]
    t0 = _unit(p[min(3,len(p)-1)]-p[0])
    t1 = _unit(p[max(0,len(p)-4)]-p[-1])
    b0,b1,b2,b3=(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3
    a=np.stack([b1[:,None]*t0,b2[:,None]*t1],axis=2).reshape(-1,2)
    rhs=(p-(b0+b1)[:,None]*p[0]-(b2+b3)[:,None]*p[-1]).reshape(-1)
    alpha=np.linalg.lstsq(a,rhs,rcond=None)[0]
    if (alpha<max(length,1)*1e-4).any() or (alpha>max(length,1)*1.5).any():
        alpha[:]=length/3
    c1,c2=p[0]+alpha[0]*t0,p[-1]+alpha[1]*t1
    fitted=b0[:,None]*p[0]+b1[:,None]*c1+b2[:,None]*c2+b3[:,None]*p[-1]
    errors=np.linalg.norm(fitted-p,axis=1)
    split=int(np.argmax(errors))
    if errors[split]<=tolerance:
        # A straight segment has no curvature and fewer editable controls.
        if length>1e-9:
            distance=np.abs(chord[0]*(p[:,1]-p[0,1])-chord[1]*(p[:,0]-p[0,0]))/length
            projection=(p-p[0])@chord/(length*length)
            if distance.max()<=tolerance*.35 and (projection>=0).all() and (projection<=1).all():
                return [(p[0],None,None,p[-1])]
        return [(p[0],c1,c2,p[-1])]
    if depth>=24:
        return [(p[i],None,None,p[i+1]) for i in range(len(p)-1)]
    split=max(1,min(len(p)-2,split))
    return _fit_span(p[:split+1],tolerance,depth+1)+_fit_span(p[split:],tolerance,depth+1)


def fit_ring(points,tolerance=.5,corner_angle=62):
    from .svgutil import rdp,mark_corners
    p=np.asarray(points,dtype=float).reshape(-1,2)
    if len(p)>1 and np.allclose(p[0],p[-1]):p=p[:-1]
    if len(p)<3:return []
    # Sparse vertices identify intentional corners; the fit uses all original samples.
    ring=np.vstack([p,p[:1]])
    simplified=rdp(ring,max(.4,tolerance))
    if np.allclose(simplified[0],simplified[-1]):simplified=simplified[:-1]
    corners=mark_corners(simplified,corner_angle)
    indexes=[]
    for point in simplified[corners]:
        indexes.append(int(np.argmin(np.sum((p-point)**2,axis=1))))
    if len(set(indexes))<2:
        start=int(np.argmin(p[:,0]))
        opposite=int(np.argmax(np.sum((p-p[start])**2,axis=1)))
        indexes=[start,opposite]
    indexes=sorted(set(indexes))
    output=[]
    for i,a in enumerate(indexes):
        b=indexes[(i+1)%len(indexes)]
        span=p[a:b+1] if b>a else np.vstack([p[a:],p[:b+1]])
        output.extend(_fit_span(span,tolerance))
    return output
