"""Editable vector export with native PDF and PostScript Level 3 gradients."""
from __future__ import annotations

import io
import re


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _load_drawing(svg_text: str):
    from svglib.svglib import svg2rlg
    # lxml 不接受带 XML 声明的 Unicode 字符串，统一用 bytes 传入
    buf = io.BytesIO(svg_text.encode("utf-8"))
    buf.name = "input.svg"
    return svg2rlg(buf)


def svg_to_pdf(svg_text: str) -> bytes:
    """Export editable paths with native PDF gradients and fill transparency."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    if '<filter' in svg_text:
        from .filter_export import expand_filters
        svg_text = expand_filters(svg_text)
    width, height, layers = _parse_svg(svg_text)
    if not width or not height:
        raise ValueError('SVG 缺少有效尺寸')
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(width, height), pageCompression=1)
    pdf.translate(0, height)
    pdf.scale(1, -1)
    for layer in layers:
        for d in layer.paths:
            path = pdf.beginPath()
            for op in _path_to_ps(d):
                parts = op.split()
                command = parts[-1]
                values = [float(value) for value in parts[:-1]]
                if command == 'moveto': path.moveTo(*values)
                elif command == 'lineto': path.lineTo(*values)
                elif command == 'curveto': path.curveTo(*values)
                elif command == 'closepath': path.close()
            pdf.saveState()
            pdf.setFillAlpha(layer.opacity)
            if layer.stroke:
                pdf.setStrokeColor(HexColor(layer.stroke))
                pdf.setStrokeAlpha(layer.opacity)
                pdf.setLineWidth(layer.stroke_width)
                pdf.setLineCap(1)
                pdf.drawPath(path,stroke=1,fill=0)
            elif layer.gradient:
                grad = layer.gradient
                stops = sorted(grad['stops'])
                colors = [HexColor(color) for _,color in stops]
                positions = [offset for offset,_ in stops]
                pdf.clipPath(path,stroke=0,fill=0,fillMode=0)
                if grad.get('type') == 'radialGradient':
                    pdf.radialGradient(grad['cx'],grad['cy'],grad['r'],colors,positions,extend=True)
                else:
                    pdf.linearGradient(grad['x1'],grad['y1'],grad['x2'],grad['y2'],colors,positions,extend=True)
            else:
                pdf.setFillColor(HexColor(layer.fill))
                pdf.setFillAlpha(layer.opacity)
                pdf.drawPath(path,stroke=0,fill=1,fillMode=0)
            pdf.restoreState()
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SVG 结构解析（导出共用）
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")


class _Layer:
    def __init__(self, fill: str, opacity: float, paths: list[str], gradient: dict | None,
                 stroke=None,stroke_width=1.):
        self.fill = fill
        self.opacity = opacity
        self.paths = paths
        self.gradient = gradient
        self.stroke=stroke
        self.stroke_width=stroke_width


def _parse_svg(svg_text: str) -> tuple[int, int, list[_Layer]]:
    size_w = re.search(r'width="([\d.]+)"', svg_text)
    size_h = re.search(r'height="([\d.]+)"', svg_text)
    width = int(float(size_w.group(1))) if size_w else 0
    height = int(float(size_h.group(1))) if size_h else 0

    from .pathparse import parse_svg_layers, SvgRaster, translate_path
    grads, layers_raw = parse_svg_layers(svg_text, transform_of=SvgRaster._transform_offset)
    layers: list[_Layer] = []
    for item in layers_raw:
        fill = item["fill"]
        grad = None
        if fill.startswith("url("):
            gid = fill[5:].rstrip(")").lstrip("#")
            grad = grads.get(gid)
            if grad and grad.get("stops"):
                fill = grad["stops"][len(grad["stops"]) // 2][1]
            else:
                fill = "#000000"
        layers.append(_Layer(fill=fill, opacity=item["opacity"],
                             paths=[translate_path(d, off) for d, off in item['d']], gradient=grad,
                             stroke=item.get('stroke'),stroke_width=item.get('stroke_width',1.)))
    return width, height, layers


def _hex_to_rgb_floats(value: str) -> tuple[float, float, float]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(ch * 2 for ch in v)
    if len(v) < 6:
        return (0.0, 0.0, 0.0)
    return int(v[0:2], 16) / 255.0, int(v[2:4], 16) / 255.0, int(v[4:6], 16) / 255.0


def _path_to_ps(d: str, eps: float = 0.01) -> list[str]:
    """把 SVG path 的 d 转成 PostScript 路径绘制命令（保留三次贝塞尔）。"""
    out: list[str] = []
    tokens = _TOKEN.findall(d)
    cmd = ""
    cur = [0.0, 0.0]
    start = [0.0, 0.0]
    i = 0

    def num() -> float | None:
        nonlocal i
        if i < len(tokens) and tokens[i][0] == "":
            val = float(tokens[i][1])
            i += 1
            return val
        return None

    def point(relative: bool, x: float, y: float) -> tuple[float, float]:
        if relative:
            return cur[0] + x, cur[1] + y
        return x, y

    while i < len(tokens):
        kind, value = tokens[i]
        if kind:
            cmd = kind
            i += 1
            if cmd in "Zz":
                out.append("closepath")
                cur = list(start)
                continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = num(), num()
            if x is None or y is None:
                break
            cur = list(point(rel, x, y))
            start = list(cur)
            out.append(f"{cur[0]:.2f} {cur[1]:.2f} moveto")
            cmd = "l" if rel else "L"
        elif c == "L":
            x, y = num(), num()
            if x is None or y is None:
                break
            cur = list(point(rel, x, y))
            out.append(f"{cur[0]:.2f} {cur[1]:.2f} lineto")
        elif c == "H":
            x = num()
            if x is None:
                break
            cur[0] = cur[0] + x if rel else x
            out.append(f"{cur[0]:.2f} {cur[1]:.2f} lineto")
        elif c == "V":
            y = num()
            if y is None:
                break
            cur[1] = cur[1] + y if rel else y
            out.append(f"{cur[0]:.2f} {cur[1]:.2f} lineto")
        elif c == "C":
            vals = [num() for _ in range(6)]
            if any(v is None for v in vals):
                break
            p1 = point(rel, vals[0], vals[1])
            p2 = point(rel, vals[2], vals[3])
            p3 = point(rel, vals[4], vals[5])
            cur = list(p3)
            out.append("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} curveto".format(
                p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]))
        else:
            raise ValueError(f'导出不支持路径指令 {cmd}，请保留 SVG')
        _ = eps
    return out


def _gradient_bands(grad: dict, width: int, height: int, levels: int = 12) -> list[tuple[str, tuple[float, float, float]]]:
    """把线性渐变拆成若干平行四边形色带，返回 [(ps 命令, 颜色)]。"""
    x1, y1, x2, y2 = grad["x1"], grad["y1"], grad["x2"], grad["y2"]
    stops = sorted(grad["stops"], key=lambda s: s[0])
    if len(stops) < 2:
        return []
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return []
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    reach = float(max(width, height) * 2 + 10)
    bands = []
    for i in range(levels):
        t0 = i / levels
        t1 = (i + 1) / levels
        tm = (t0 + t1) / 2
        # 插值取色
        color = _sample_stops(stops, tm)
        p0 = (x1 + ux * length * t0, y1 + uy * length * t0)
        p1 = (x1 + ux * length * t1, y1 + uy * length * t1)
        quad = [
            (p0[0] + nx * reach, p0[1] + ny * reach),
            (p1[0] + nx * reach, p1[1] + ny * reach),
            (p1[0] - nx * reach, p1[1] - ny * reach),
            (p0[0] - nx * reach, p0[1] - ny * reach),
        ]
        ps = (f"{quad[0][0]:.2f} {quad[0][1]:.2f} moveto "
              f"{quad[1][0]:.2f} {quad[1][1]:.2f} lineto "
              f"{quad[2][0]:.2f} {quad[2][1]:.2f} lineto "
              f"{quad[3][0]:.2f} {quad[3][1]:.2f} lineto closepath")
        bands.append((ps, color))
    return bands


def _sample_stops(stops: list[tuple[float, str]], t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))
    if not stops:
        raise ValueError('渐变缺少色标')
    if t <= stops[0][0]:
        return _hex_to_rgb_floats(stops[0][1])
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        if a[0] <= t <= b[0]:
            span = max(1e-6, b[0] - a[0])
            k = (t - a[0]) / span
            ca, cb = _hex_to_rgb_floats(a[1]), _hex_to_rgb_floats(b[1])
            return tuple(ca[j] + (cb[j] - ca[j]) * k for j in range(3))
    return _hex_to_rgb_floats(stops[-1][1])


def svg_to_eps(svg_text: str, gradient_levels: int = 12) -> bytes:
    """Export editable paths with native Level 3 linear and radial shading."""
    from .eps_export import export_eps
    return export_eps(svg_text)


def validate_export(payload: bytes, fmt: str) -> dict:
    """导出产物校验：结构头尾 + 可解析性，返回 {ok, checks, detail}。

    PDF/EPS 用头尾与关键结构验证；SVG 用 XML 解析器验证合法可解析。
    """
    fmt = (fmt or "svg").lower()
    checks: dict[str, bool] = {}
    detail = ""
    if fmt in ("pdf", "ai"):
        checks["header"] = payload[:5] == b"%PDF-"
        tail = payload[-2048:].decode("latin-1", errors="ignore")
        checks["eof"] = "%%EOF" in tail
        checks["size_sane"] = len(payload) > 800
    elif fmt == "eps":
        head = payload[:32].decode("latin-1", errors="ignore")
        checks["header"] = head.startswith("%!PS-Adobe-")
        tail = payload[-2048:].decode("latin-1", errors="ignore")
        checks["eof"] = "%%EOF" in tail
        checks["size_sane"] = len(payload) > 200
    else:
        import xml.etree.ElementTree as ET
        try:
            ET.fromstring(payload)
            checks["xml_well_formed"] = True
        except ET.ParseError as exc:
            checks["xml_well_formed"] = False
            detail = str(exc)
    checks["passed"] = all(checks.values()) if checks else False
    return {"checks": checks, "detail": detail, "fmt": fmt,
            "bytes": len(payload)}


def minify_svg(svg_text: str) -> str:
    text = re.sub(r">\s+<", "><", svg_text).strip()
    return text
