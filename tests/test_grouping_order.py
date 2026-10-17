"""字级分组应保持渲染结果与路径完整性。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.pipeline.svgutil import SvgDocument, SvgLayer
from app.pipeline.pathparse import SvgRaster


def _crossing_document():
    doc=SvgDocument(width=100,height=100)
    doc.layers=[SvgLayer(color='#202030',paths=['M0 0L100 0L100 100L0 100Z']),
                SvgLayer(color='#ff0000',paths=['M10 10L40 10L40 90L10 90Z']),
                SvgLayer(color='#0000ff',paths=['M60 10L90 10L90 90L60 90Z'])]
    return doc


def test_grouping_cannot_cover_another_glyph_fill():
    doc=_crossing_document()
    flat=SvgRaster(doc.to_svg()).rasterize()
    # 跨字底色归入第二字时，逐字输出会在第一字填色之后重画底色。
    doc.glyph_groups=[[(1,0)],[(0,0),(2,0)]]
    assert not doc._glyph_paint_order_is_safe()
    svg=doc.to_svg()
    assert 'id="layer-1"' in svg
    assert np.array_equal(flat,SvgRaster(svg).rasterize())


def test_separate_glyphs_keep_editable_glyph_groups():
    doc=SvgDocument(width=100,height=100)
    doc.layers=[SvgLayer(color='#202030',paths=['M0 0L40 0L40 100L0 100Z','M60 0L100 0L100 100L60 100Z']),
                SvgLayer(color='#ff0000',paths=['M10 10L30 10L30 90L10 90Z','M70 10L90 10L90 90L70 90Z'])]
    flat=SvgRaster(doc.to_svg()).rasterize()
    doc.glyph_groups=[[(0,0),(1,0)],[(0,1),(1,1)]]
    assert doc._glyph_paint_order_is_safe()
    svg=doc.to_svg()
    assert 'id="glyph-1"' in svg and 'id="glyph-2"' in svg
    assert np.array_equal(flat,SvgRaster(svg).rasterize())


def test_missing_or_duplicate_group_path_uses_complete_layers():
    doc=_crossing_document()
    flat=SvgRaster(doc.to_svg()).rasterize()
    for groups in [[[(1,0)],[(2,0)]],[[(0,0),(1,0)],[(0,0),(2,0)]]]:
        doc.glyph_groups=groups
        assert not doc._glyph_paint_order_is_safe()
        assert np.array_equal(flat,SvgRaster(doc.to_svg()).rasterize())


def test_group_opacity_is_applied_to_the_whole_layer():
    doc=SvgDocument(width=100,height=100)
    doc.layers=[SvgLayer(color='#ff0000',opacity=.5,
                        paths=['M0 0L70 0L70 100L0 100Z','M30 0L100 0L100 100L30 100Z'])]
    doc.glyph_groups=[[(0,0)],[(0,1)]]
    assert not doc._glyph_paint_order_is_safe()
    svg=doc.to_svg()
    assert svg.count('opacity="0.500"')==1
    assert 'id="layer-1"' in svg


def test_control_hull_overlap_and_filters_keep_layer_order():
    doc=_crossing_document()
    doc.layers[1].paths=['M10 10C150 10 150 90 10 90Z']
    doc.glyph_groups=[[(1,0)],[(0,0),(2,0)]]
    assert not doc._glyph_paint_order_is_safe()
    doc.glyph_groups=[[(0,0),(1,0),(2,0)]]
    doc.layers[1].filter_id='shadow-soft'
    assert not doc._glyph_paint_order_is_safe()
    assert 'filter="url(#shadow-soft)"' in doc.to_svg()
