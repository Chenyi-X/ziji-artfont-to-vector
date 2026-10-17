"""原始像素、预处理、SVG 渲染及掩膜差异预览。"""
from __future__ import annotations
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .pathparse import SvgRaster
from .preprocess import PreparedImage

FONT_CANDIDATES = ['C:/Windows/Fonts/msyhbd.ttc', 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']


def _font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def checkerboard(h, w, cell=16):
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy//cell + xx//cell) % 2) == 0, 235, 205).astype(np.uint8)
    return np.dstack([board, board, board])


def over_checker(rgba, input_bgr=True):
    a = rgba[:, :, 3:4].astype(np.float32) / 255
    rgb = rgba[:, :, :3][:, :, ::-1] if input_bgr else rgba[:, :, :3]
    return (checkerboard(*rgba.shape[:2]) * (1-a) + rgb * a).astype(np.uint8)


def diff_overlay(orig_mask, out_mask):
    canvas = np.full((*orig_mask.shape, 3), 245, np.uint8)
    canvas[orig_mask & out_mask] = (120, 120, 120)
    canvas[orig_mask & ~out_mask] = (60, 190, 90)
    canvas[out_mask & ~orig_mask] = (230, 70, 200)
    return canvas


def svg_preview_rgba(svg_text):
    return SvgRaster(svg_text).rasterize()


def build_comparison(prepared: PreparedImage, svg_text: str, title='', elapsed_ms=None,
                     max_side=1200, raster=None):
    """原始像素仅经过与输出相同的缩放、旋转和裁切；保留坐标变换供追溯。"""
    rendered = raster if raster is not None else svg_preview_rgba(svg_text)
    raw = prepared.source_rgba
    if raw is None:
        raise ValueError('对比图需要未经增强的原始像素')
    panel_width = max(120, max_side//4)
    scale = min(1., (panel_width-16)/max(1, rendered.shape[1]))
    size = (max(1, round(rendered.shape[1]*scale)), max(1, round(rendered.shape[0]*scale)))
    raw_small = over_checker(cv2.resize(raw, size, interpolation=cv2.INTER_AREA), input_bgr=False)
    processed = cv2.resize(prepared.rgba, size, interpolation=cv2.INTER_AREA)
    output = cv2.resize(rendered, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(prepared.mask, size, interpolation=cv2.INTER_NEAREST)>127
    panels = [raw_small, over_checker(processed), over_checker(output, input_bgr=False),
              diff_overlay(mask, output[:, :, 3]>127)]
    labels = ['原始像素（仅几何对齐）', '预处理结果', 'SVG 内部预览', '掩膜差异（绿原 / 紫输出）']
    canvas = Image.new('RGB', (panel_width*4, size[1]+92), 'white')
    draw = ImageDraw.Draw(canvas)
    head = title or 'art-text vectorize'
    if elapsed_ms is not None:
        head += f' ({elapsed_ms:.0f} ms)'
    draw.text((12, 10), head, font=_font(22), fill=(20,20,20))
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = i*panel_width
        draw.text((x+8, 50), label, font=_font(17), fill=(90,90,90))
        canvas.paste(Image.fromarray(panel), (x+8, 88))
    return np.asarray(canvas)


def encode_png(img_rgb):
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format='PNG', optimize=True)
    return buf.getvalue()
