from __future__ import annotations
import base64
import io
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from app.pipeline.assessment import assess_result
from app.pipeline.preprocess import preprocess
from app.pipeline.service import convert_bytes


def test_missing_measurements_are_untested():
    a=assess_result({}, {}, None, (1000,1000))
    assert a['status']=='untested'
    assert a['items']['node_average']['status']=='untested'
    assert a['items']['gradient_levels']['status']=='untested'
    assert a['items']['svg_structure']['status']=='untested'


def test_gradient_label_requires_check_even_when_engine_misses_it():
    m={'is_gradient_art':False,'color_levels_out':7}
    assert assess_result(m,{},None,(1000,1000),gradient_required=True)['items']['gradient_levels']['status']=='fail'
    assert assess_result(m,{},None,(1000,1000),gradient_required=False)['items']['gradient_levels']['status']=='not_applicable'


def test_strict_color_and_true_glyph_count():
    m={'color_mode':'multi','color_deviation':.04,'glyph_source':'provided','glyph_count':1,'node_per_glyph':2216}
    a=assess_result(m,{},100,(1000,1000))
    assert a['items']['color']['status']=='fail'
    assert a['items']['nodes_each_glyph']['status']=='fail'
    m['node_per_glyph']=300
    m['glyph_count']=4
    assert assess_result(m,{},100,(1000,1000))['items']['nodes_each_glyph']['status']=='untested'


def test_full_timing_includes_validation_and_preview(monkeypatch):
    import app.pipeline.service as service
    values=iter([0,1,2,3,4,5,7,8,10,11])
    monkeypatch.setattr(service.time,'perf_counter',lambda:next(values))
    img=np.full((600,600,3),255,np.uint8)
    cv2.rectangle(img,(20,20),(580,580),(20,40,90),-1)
    data=cv2.imencode('.png',img)[1].tobytes()
    res=convert_bytes('timer.png',data,with_preview=True,strict_input=False,glyph_count=1)
    assert not res.error
    assert res.elapsed_ms==11000
    assert res.timings['vectorize_ms']==1000
    assert res.timings['verify_ms']==2000
    assert res.timings['preview_ms']==2000
    assert res.metrics['assessment']['items']['single_time']['status']=='fail'
    assert res.metrics['elapsed_ms']==res.timings['total_ms']==res.to_dict()['elapsed_ms']


def test_aligned_source_keeps_original_alpha_and_colors():
    img=np.full((600,600,4),(40,80,120,0),np.uint8)
    img[30:570,30:570,3]=255
    img[30:35,30:570,3]=96
    p=preprocess(img)
    assert p.crop_box is None and p.rot_matrix is None
    assert np.array_equal(p.source_rgba,img[:,:,[2,1,0,3]])
    assert p.quality.orig_bbox_ratio>.6
    res=convert_bytes('alpha.png',cv2.imencode('.png',img)[1].tobytes(),with_preview=True,strict_input=False,glyph_count=1)
    source=np.array(Image.open(io.BytesIO(res.source_png)).convert('RGBA'))
    assert np.array_equal(source,p.source_rgba)


def test_small_input_strictly_rejected():
    img=np.full((400,400,3),255,np.uint8)
    cv2.rectangle(img,(20,20),(380,380),(0,0,0),-1)
    res=convert_bytes('small.png',cv2.imencode('.png',img)[1].tobytes(),strict_input=True)
    assert res.error and not res.svg


def test_low_occupancy_stays_invalid_after_crop():
    img=np.full((600,600,3),255,np.uint8)
    cv2.rectangle(img,(200,200),(400,400),(0,0,0),-1)
    p=preprocess(img)
    assert p.quality.autocropped
    assert p.quality.invalid and 'subject_small' in p.quality.invalid_reasons
    assert p.quality.orig_bbox_ratio<.6<p.quality.subject_bbox_ratio
