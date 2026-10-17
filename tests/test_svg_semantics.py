"""Regression cases for SVG paint inheritance, translations and native exports."""
import numpy as np
import pytest
from app.pipeline.pathparse import SvgRaster,parse_svg_layers,translate_path,parse_path
from app.pipeline.export import svg_to_pdf,svg_to_eps


def wrap(body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="100">{body}</svg>'


def test_path_fill_is_preserved():
    raster=SvgRaster(wrap('<path fill="#ff0000" d="M10 10 L40 10 L40 40 L10 40 Z"/>'))
    pixel=raster.rasterize()[20,20]
    assert tuple(pixel[:3])==(255,0,0)
    assert pixel[3]==255


def test_nested_fill_inherits_and_keeps_paint_order():
    text=wrap('<g fill="#ff0000"><path d="M0 0H80V80H0Z"/><g fill="#00ff00"><path d="M0 0H60V60H0Z"/></g><path d="M0 0H20V20H0Z"/></g>')
    pixels=SvgRaster(text).rasterize()
    assert tuple(pixels[10,10,:3])==(255,0,0)
    assert tuple(pixels[40,40,:3])==(0,255,0)
    assert tuple(pixels[70,70,:3])==(255,0,0)


def test_translate_is_not_scaled_twice():
    text=wrap('<g transform="translate(20 10)" fill="#ff0000"><path transform="translate(5 5)" d="M0 0H20V20H0Z"/></g>')
    raster=SvgRaster(text)
    for scale in [1,2,4]:
        pixels=raster.rasterize(scale=scale)
        assert pixels[20,30,3]>200
        assert pixels[5,5,3]==0
        assert pixels[70,100,3]==0


@pytest.mark.parametrize('d',['M1 2 C3 4 5 6 8 9 Z','m1 2 c3 4 5 6 8 9 z','M1 2H8V12L1 12Z','M1 2Q8 3 10 9T1 2Z'])
def test_translate_preserves_sampled_geometry(d):
    expected=[p+np.array([12.,19.]) for p in parse_path(d)]
    actual=parse_path(translate_path(d,(12.,19.)))
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-5)


def test_pdf_preserves_group_opacity_and_holes():
    import fitz
    text=wrap('<g fill="#ff0000" opacity="0.4"><path d="M0 0H100V90H0Z M30 30H60V60H30Z"/></g>')
    doc=fitz.open(stream=svg_to_pdf(text),filetype='pdf')
    pix=doc[0].get_pixmap(alpha=True)
    rgba=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,4)
    assert 95<=rgba[10,10,3]<=105
    assert rgba[40,40,3]==0
    with pytest.raises(ValueError,match='透明度'):svg_to_eps(text)


def test_radial_eps_uses_native_shading():
    text=wrap('<defs><radialGradient id="g" cx="50" cy="50" r="40"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#000000"/></radialGradient></defs><g fill="url(#g)"><path d="M0 0H100V90H0Z"/></g>')
    eps=svg_to_eps(text)
    assert b'/ShadingType 3' in eps
    assert b'/FunctionType 3' in eps
    assert b'eoclip' in eps
