"""Additional upload, naming, cache and API contract regressions."""
import io,json,zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app,_sample_cache_key
from app.pipeline import service
from app.pipeline.service import ConvertResult,convert_bytes,inspect_bytes,batch_convert
from app.pipeline.vectorize import VectorizeOptions
client=TestClient(app)


def test_empty_image_is_reported():
    assert convert_bytes('empty.png',b'').error
    assert inspect_bytes('empty.png',b'')['problems']


def test_reserved_batch_stem_terminates_and_keeps_summary(monkeypatch):
    monkeypatch.setattr(service,'convert_bytes',lambda *a,**kw:ConvertResult('_batch_summary.png','_batch_summary','<svg/>',{}, {}, {}, {},1,[]))
    payload,_=batch_convert([('_batch_summary.png',b'x')])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert 'image_batch_summary.svg' in archive.namelist()
        assert len(archive.namelist())==len(set(archive.namelist()))
        assert json.loads(archive.read('_batch_summary.json'))['count']==1


@pytest.mark.parametrize('options',[{'max_nodes':501},{'gradient_levels':7},{'preset':'unknown'},{'engine':'unknown'},{'tolerance':'nan'}])
def test_api_invalid_options_return_422(options):
    response=client.post('/api/convert',files={'file':('x.png',b'x','image/png')},data=options)
    assert response.status_code==422


def test_api_oversize_returns_413():
    response=client.post('/api/inspect',files={'file':('large.png',b'x'*(5*1024*1024+1),'image/png')})
    assert response.status_code==413


def test_api_nonascii_export_name():
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><g fill="#123456"><path d="M0 0H20V20H0Z"/></g></svg>'
    response=client.post('/api/export',data={'svg':svg,'fmt':'pdf','stem':'艺术字'})
    assert response.status_code==200
    assert "filename*=UTF-8''" in response.headers['content-disposition']
    assert response.content.startswith(b'%PDF-')


def test_api_unsupported_export_is_explicit():
    response=client.post('/api/export',data={'svg':'<svg/>','fmt':'jpeg'})
    assert response.status_code==422


def test_cache_includes_all_output_options():
    opts=VectorizeOptions()
    keys=[_sample_cache_key('13_cartoon.png',opts,False,True,4),
          _sample_cache_key('13_cartoon.png',opts,False,False,4),
          _sample_cache_key('13_cartoon.png',opts,False,True,5)]
    assert len(set(keys))==3
