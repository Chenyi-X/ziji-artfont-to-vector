"""Optional official Vectorizer.AI API adapter; credentials stay on the server."""
from __future__ import annotations
import os
import time
from pathlib import Path
import base64
import json
import urllib.request
import urllib.error
import uuid


def provider_status():
    ready=bool(os.environ.get('VECTORIZER_API_ID') and os.environ.get('VECTORIZER_API_SECRET'))
    return {'id':'vectorizer_ai','name':'Vectorizer.AI','configured':ready,
            'mode':os.environ.get('VECTORIZER_API_MODE','test'),
            'documentation':'https://vectorizer.ai/api/documentation'}


def vectorize_remote(filename,data,max_colors=0):
    status=provider_status()
    if not status['configured']:
        raise ValueError('请在服务环境配置 VECTORIZER_API_ID 和 VECTORIZER_API_SECRET 后重启')
    if status['mode'] not in {'test','production'}:
        raise ValueError('VECTORIZER_API_MODE 必须为 test 或 production')
    boundary='vector-'+uuid.uuid4().hex
    options={'mode':status['mode'],'output.file_format':'svg','output.svg.fixed_size':'true',
             'output.svg.adobe_compatibility_mode':'true','output.parameterized_shapes.flatten':'true',
             'output.curves.allowed.quadratic_bezier':'false','output.curves.allowed.circular_arc':'false',
             'output.curves.allowed.elliptical_arc':'false','output.gap_filler.enabled':'false',
             'output.shape_stacking':'stacked','output.group_by':'color','processing.max_colors':str(max_colors)}
    parts=[]
    for key,value in options.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    ext=Path(filename).suffix.lower();safe='image'+(ext if ext in {'.png','.jpg','.jpeg','.webp'} else '.png')
    parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{safe}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(),data,f'\r\n--{boundary}--\r\n'.encode()])
    credentials=(os.environ['VECTORIZER_API_ID']+':'+os.environ['VECTORIZER_API_SECRET']).encode()
    request=urllib.request.Request('https://vectorizer.ai/api/v1/vectorize',data=b''.join(parts),method='POST',
        headers={'Authorization':'Basic '+base64.b64encode(credentials).decode(),
                 'Content-Type':'multipart/form-data; boundary='+boundary})
    started=time.perf_counter()
    try:
        with urllib.request.urlopen(request,timeout=90) as response:
            payload=response.read(20*1024*1024+1)
            credits=response.headers.get('X-Credits-Charged')
    except urllib.error.HTTPError as exc:
        raise ValueError(f'第三方服务返回 HTTP {exc.code}，请检查凭据、订阅或额度') from exc
    if len(payload)>20*1024*1024 or b'<svg' not in payload[:4096]:
        raise ValueError('第三方返回的文件不是有效 SVG 或超过大小限制')
    from defusedxml.ElementTree import fromstring
    root=fromstring(payload)
    for node in root.iter():
        tag=node.tag.rsplit('}',1)[-1]
        if tag in {'script','foreignObject','image','iframe','object','embed','style'}:
            raise ValueError('第三方 SVG 含不允许的内容')
        for key,value in node.attrib.items():
            local=key.rsplit('}',1)[-1].lower()
            if local.startswith('on') or local=='style' or (local=='href' and not value.startswith('#')):
                raise ValueError('第三方 SVG 含不允许的外部资源或事件')
            if 'url(' in value and 'url(#' not in value:
                raise ValueError('第三方 SVG 含不允许的外部资源')
    svg=payload.decode('utf-8-sig')
    return svg,{'provider':status['name'],'mode':status['mode'],'watermarked':status['mode']=='test',
                'credits_charged':credits,'elapsed_ms':round((time.perf_counter()-started)*1000,1)}
