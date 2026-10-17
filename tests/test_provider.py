"""Official provider contract, authentication redaction and explicit test mode."""
import io
import urllib.error
import pytest
from app.providers import provider_status,vectorize_remote


def test_provider_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv('VECTORIZER_API_ID',raising=False)
    monkeypatch.delenv('VECTORIZER_API_SECRET',raising=False)
    assert not provider_status()['configured']
    with pytest.raises(ValueError,match='VECTORIZER_API_ID'):
        vectorize_remote('sample.png',b'png')


def test_provider_uses_official_authenticated_api(monkeypatch):
    monkeypatch.setenv('VECTORIZER_API_ID','test-id')
    monkeypatch.setenv('VECTORIZER_API_SECRET','test-secret')
    monkeypatch.setenv('VECTORIZER_API_MODE','test')
    class Response(io.BytesIO):
        headers={'X-Credits-Charged':'0'}
    def request(req,timeout):
        assert req.full_url=='https://vectorizer.ai/api/v1/vectorize'
        assert req.headers['Authorization'].startswith('Basic ')
        assert b'name="mode"\r\n\r\ntest' in req.data
        return Response(b'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><path d="M0 0H20V20Z"/></svg>')
    monkeypatch.setattr('urllib.request.urlopen',request)
    svg,detail=vectorize_remote('sample.png',b'png')
    assert detail['watermarked']
    assert 'test-secret' not in str(provider_status())+str(detail)
    assert '<svg' in svg


@pytest.mark.parametrize('body', [
    '<path onload="alert(1)" d="M0 0H20V20Z"/>',
    '<use href="https://example.invalid/a.svg#x"/>',
    '<path fill="url(https://example.invalid/x)" d="M0 0H20V20Z"/>',
])
def test_provider_rejects_active_or_external_content(monkeypatch,body):
    monkeypatch.setenv('VECTORIZER_API_ID','test-id')
    monkeypatch.setenv('VECTORIZER_API_SECRET','test-secret')
    class Response(io.BytesIO):headers={}
    payload=('<svg xmlns="http://www.w3.org/2000/svg">'+body+'</svg>').encode()
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Response(payload))
    with pytest.raises(ValueError,match='不允许'):
        vectorize_remote('sample.png',b'png')


def test_provider_rejects_embedded_bitmap(monkeypatch):
    monkeypatch.setenv('VECTORIZER_API_ID','test-id')
    monkeypatch.setenv('VECTORIZER_API_SECRET','test-secret')
    class Response(io.BytesIO):headers={}
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Response(b'<svg xmlns="http://www.w3.org/2000/svg"><image href="data:image/png;base64,AA=="/></svg>'))
    with pytest.raises(ValueError,match='不允许'):
        vectorize_remote('sample.png',b'png')
