"""Independent renderer and source-boundary regressions for development release."""
from pathlib import Path
import io
import cv2
import numpy as np
import fitz
from app.pipeline.pathparse import SvgRaster
from app.pipeline.export import svg_to_pdf
from app.pipeline.preprocess import preprocess
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions

ROOT=Path(__file__).resolve().parents[1]


def test_translucent_rgba_is_straight_and_composites_once():
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"><path fill="#ff0000" opacity="0.5" d="M0 0H40V40H0Z"/></svg>'
    pixels=SvgRaster(svg).rasterize()
    assert tuple(pixels[20,20,:3])==(255,0,0)
    assert 126<=pixels[20,20,3]<=129
    a=pixels[20,20,3]/255
    painted=pixels[20,20,:3]*a+255*(1-a)
    np.testing.assert_allclose(painted,[255,127,127],atol=2)


def test_clean_lettering_mask_keeps_original_thin_features():
    image=np.full((600,600,3),255,np.uint8)
    cv2.rectangle(image,(30,30),(570,570),(0,0,0),2)
    image[200:203,100:500]=0
    prepared=preprocess(image)
    assert prepared.rot_matrix is None
    reference=np.linalg.norm(image[:,:,::-1].astype(float)-255,axis=2)>20
    np.testing.assert_array_equal(prepared.mask>127,reference)


def test_kai_and_handwriting_restore_original_contours():
    for name in ['03_kai.png','07_shuti_hand.png']:
        result=convert_bytes(name,(ROOT/'samples/images'/name).read_bytes(),
                             VectorizeOptions(glyph_count=4),strict_input=False,with_preview=False)
        assert result.stats.get('source_silhouette')
        assert result.metrics['shape_iou']>=.97
        assert 50<=result.metrics['node_per_glyph']<=500


def test_gaussian_shadow_pdf_remains_vector_and_visually_close():
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><defs><filter id="blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3"/></filter></defs><g fill="#246080" filter="url(#blur)"><path d="M20 20H80V80H20Z"/></g></svg>'''
    rgba=SvgRaster(svg).rasterize();alpha=rgba[:,:,3:4]/255
    reference=rgba[:,:,:3]*alpha+255*(1-alpha)
    with fitz.open(stream=svg_to_pdf(svg),filetype='pdf') as doc:
        assert not doc[0].get_images()
        assert doc[0].get_drawings()
        pix=doc[0].get_pixmap(alpha=False)
        pixels=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
    assert np.abs(reference-pixels.astype(float)).mean()/255<.02
