"""Job capacity, cancellation, archive identity and report aggregation tests."""
from __future__ import annotations
import io,json,threading,time,zipfile
import numpy as np
import cv2
import pytest
from app.jobs import JobManager,JobQueueFull
from app.pipeline.service import batch_convert,ConvertResult,convert_bytes
from app.pipeline.vectorize import VectorizeOptions
from eval.regression_req import aggregate_status


def wait(manager,job_id):
    end=time.monotonic()+3
    while time.monotonic()<end:
        result=manager.get(job_id)
        if result['status'] in {'done','error','cancelled'}:return result
        time.sleep(.005)
    raise AssertionError('job did not complete')


def test_job_capacity_and_active_retention():
    manager=JobManager(workers=1,capacity=2,retained=1)
    started=threading.Event();release=threading.Event()
    def run(progress):
        started.set();release.wait(2);progress('verify');return {'svg':'ok'}
    try:
        first=manager.submit(run);assert started.wait(1)
        second=manager.submit(lambda cb:{'svg':'second'})
        with pytest.raises(JobQueueFull):manager.submit(run)
        assert manager.get(first)['status']=='running'
        assert manager.get(second)['status']=='queued'
        manager.cancel(second);release.set()
        assert wait(manager,second)['status']=='cancelled'
    finally:
        release.set();manager.shutdown()


def test_error_payload_is_terminal_error():
    manager=JobManager(workers=1,capacity=1)
    try:
        job=manager.submit(lambda cb:{'error':'invalid image'})
        assert wait(manager,job)['status']=='error'
        job=manager.submit(lambda cb:{'svg':'valid'})
        assert wait(manager,job)['status']=='done'
    finally:manager.shutdown()


def test_terminal_cancel_does_not_change_success():
    manager=JobManager(workers=1,capacity=1)
    try:
        job=manager.submit(lambda cb:{'svg':'ok'})
        assert wait(manager,job)['status']=='done'
        assert manager.cancel(job)['status']=='done'
    finally:manager.shutdown()


def test_duplicate_batch_names_and_traceability(monkeypatch):
    from app.pipeline import service
    def fake(name,data,*args,**kwargs):
        return ConvertResult(name,'same','<svg/>',{}, {},{}, {},1,[])
    monkeypatch.setattr(service,'convert_bytes',fake)
    payload,results=batch_convert([('a/same.png',b'a'),('b/same.jpg',b'b'),('same.png',b'c')])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names=archive.namelist()
        assert len(names)==len(set(names))
        assert {'same.svg','same__2.svg','same__3.svg'}<=set(names)
        summary=json.loads(archive.read('_batch_summary.json'))
        assert [r['source'] for r in summary['files']]==['a/same.png','b/same.jpg','same.png']
        assert summary['options']['max_nodes']==500
        assert summary['wall_ms']>=0


def test_batch_exception_preserves_other_results(monkeypatch):
    from app.pipeline import service
    def fake(name,data,*args,**kwargs):
        if name=='bad.png':raise RuntimeError('decode')
        return ConvertResult(name,'good','<svg/>',{}, {},{}, {},1,[])
    monkeypatch.setattr(service,'convert_bytes',fake)
    payload,results=batch_convert([('bad.png',b'x'),('good.png',b'x')])
    assert results[0].error
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert 'good.svg' in archive.namelist()
        assert 'bad.error.txt' in archive.namelist()


def test_repeat_failures_are_aggregated():
    assert aggregate_status([{'status':'pass'},{'status':'fail'}])=='fail'
    assert aggregate_status([{'status':'pass'},{'status':'untested'}])=='untested'
    assert aggregate_status([{'status':'pass'}])=='pass'
    assert aggregate_status([])=='untested'


def test_grayscale_png_decodes():
    image=np.full((500,500),255,np.uint8)
    cv2.putText(image,'TEST',(20,270),cv2.FONT_HERSHEY_SIMPLEX,3,0,10)
    ok,data=cv2.imencode('.png',image);assert ok
    result=convert_bytes('gray.png',data.tobytes(),with_preview=False,strict_input=False,glyph_count=4)
    assert not result.error
    assert '<svg' in result.svg


def test_options_glyph_count_reaches_metrics():
    image=np.full((500,500,3),255,np.uint8)
    cv2.putText(image,'TEST',(20,270),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10)
    ok,data=cv2.imencode('.png',image);assert ok
    result=convert_bytes('text.png',data.tobytes(),VectorizeOptions(glyph_count=4),with_preview=False,strict_input=False)
    assert result.metrics['glyph_count']==4
    assert result.metrics['glyph_source']=='provided'
