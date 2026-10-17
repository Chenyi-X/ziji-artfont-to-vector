"""Native rounded strokes stay editable and contribute their actual anchors."""
import numpy as np
from app.pipeline.svgutil import SvgDocument,SvgLayer
from app.pipeline.pathparse import SvgRaster
from app.pipeline.export import svg_to_pdf,svg_to_eps


def test_round_stroke_counts_anchors_and_renders():
    doc=SvgDocument(100,100,[SvgLayer('none',['M20 50 L80 50'],stroke='#123456',stroke_width=8)])
    svg=doc.to_svg();raster=SvgRaster(svg)
    assert raster.node_stats()['node_total']==2
    pixels=raster.rasterize()
    assert pixels[50,50,3]==255
    assert tuple(pixels[50,50,:3])==(18,52,86)
    assert pixels[30,50,3]==0
    assert svg_to_pdf(svg).startswith(b'%PDF-')
    assert b'8.000000 setlinewidth 1 setlinecap stroke' in svg_to_eps(svg)


def test_grouped_stroke_keeps_style():
    doc=SvgDocument(100,100,[SvgLayer('none',['M20 50 L80 50'],stroke='#123456',stroke_width=8)])
    doc.glyph_groups=[[(0,0)]]
    svg=doc.to_svg()
    assert 'id="glyph-1"' in svg
    assert 'stroke="#123456"' in svg
    assert SvgRaster(svg).node_stats()['node_total']==2
