"""Persistent resvg renderer for SVG filters, gradients and straight RGBA pixels."""
from __future__ import annotations
import atexit
import base64
import json
from pathlib import Path
import shutil
import subprocess
import threading

import cv2
import numpy as np

_LOCK = threading.RLock()
_PROCESS = None


def shutdown():
    global _PROCESS
    with _LOCK:
        if _PROCESS is not None:
            _PROCESS.terminate()
            _PROCESS.wait(timeout=5)
            _PROCESS = None


atexit.register(shutdown)


def render_svg(svg: str, background=(0, 0, 0, 0)) -> np.ndarray:
    global _PROCESS
    worker = Path(__file__).resolve().parents[2] / 'eval/tools/render_worker.mjs'
    with _LOCK:
        if _PROCESS is None or _PROCESS.poll() is not None:
            node = shutil.which('node')
            if node is None:
                raise RuntimeError('SVG 渲染需要 Node.js 22+ 和 @resvg/resvg-js，请参照部署文档安装')
            _PROCESS = subprocess.Popen(
                [node, str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        request = {'svg': svg, 'background': 'rgba(%d,%d,%d,%.6f)' % (
            *background[:3], background[3] / 255)}
        _PROCESS.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
        _PROCESS.stdin.flush()
        line = _PROCESS.stdout.readline()
        if not line:
            raise RuntimeError('SVG 渲染进程意外退出')
        result = json.loads(line)
        if 'error' in result:
            raise ValueError('SVG 渲染失败 ' + result['error'])
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(result['png']), np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise RuntimeError('SVG 渲染没有返回有效 PNG')
    if pixels.shape[2] == 3:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2BGRA)
    return cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGBA)
