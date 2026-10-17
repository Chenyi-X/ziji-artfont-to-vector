"""Independent regressions for source-edge improvements and texture rejection."""
from pathlib import Path
import cv2
import numpy as np
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions
from app.pipeline.preprocess import preprocess
from app.pipeline.pathparse import SvgRaster
from app.pipeline.candidate_guard import independent_raster
from app.pipeline.topology import compare_topology

ROOT=Path(__file__).resolve().parent.parent


def test_source_edges_restore_serif_without_closing_gaps():
    name='02_song_serif.png';data=(ROOT/'samples/images'/name).read_bytes()
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
    prepared=preprocess(image)
    # Read SVG with a separate PDF renderer for the geometry comparison.
    from app.pipeline.export import svg_to_pdf
    import fitz
    opts=VectorizeOptions.from_preset('balanced',glyph_count=4)
    result=convert_bytes(name,data,opts,with_preview=False,strict_input=False)
    assert result.stats.get('source_edges_selected') or result.stats.get('source_tones_selected')
    with fitz.open(stream=svg_to_pdf(result.svg),filetype='pdf') as pdf:
        pix=pdf[0].get_pixmap(alpha=True)
        raster=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,4)
    reference=np.linalg.norm(prepared.rgb_raw.astype(float)-np.asarray(prepared.bg_color),axis=2)>20
    mask=raster[:,:,3]>127
    iou=np.logical_and(reference,mask).sum()/np.logical_or(reference,mask).sum()
    assert iou>=.97
    topology=compare_topology(reference,mask)
    assert not topology['holes']['extra_regions']
    assert all(item['status']=='pass' for item in topology['holes']['regions'])


def test_compact_brush_is_rejected_when_local_texture_is_lost():
    name='14_brush_feibai.png';data=(ROOT/'samples/images'/name).read_bytes()
    result=convert_bytes(name,data,VectorizeOptions(glyph_count=1),with_preview=False,strict_input=False)
    comparison=result.stats.get('brush_comparison')
    assert comparison
    assert not comparison['texture']['passed']
    assert 'local_brush_texture' in comparison['rejected']
    assert not result.stats.get('brush_candidate')
    assert result.metrics['node_per_glyph']>500
