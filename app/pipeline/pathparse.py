"""SVG 路径解析、展平与栅格化（用于节点统计、输出复核与验收指标计算）。

支持 M/L/H/V/C/S/Q/T/A/Z 的绝对与相对写法，含隐式重复参数。
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

_TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")


def parse_path(d: str, offset: tuple[float, float] = (0.0, 0.0),
               resolution: int = 24) -> list[np.ndarray]:
    """解析 d 属性，返回若干条展平后的折线点序列（Nx2 float64）。

    offset 为整体平移量，用于消化 transform="translate(x,y)"。
    """
    tokens: list[tuple[str, str]] = []
    for m in _TOKEN.finditer(d):
        if m.group(1):
            tokens.append(("cmd", m.group(1)))
        else:
            tokens.append(("num", m.group(2)))

    subpaths: list[np.ndarray] = []
    pts: list[tuple[float, float]] = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    cmd = ""
    i = 0
    nums: list[float] = []

    def num(idx: int) -> float:
        return float(tokens[idx][1])

    def flush() -> None:
        nonlocal pts
        if len(pts) >= 2:
            subpaths.append(np.asarray(pts, dtype=np.float64))
        pts = []

    def bezier(p0, p1, p2, p3, n=resolution):
        t = np.linspace(0, 1, n)[:, None]
        p = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
        return p

    def quad(p0, p1, p2, n=max(8, resolution - 4)):
        t = np.linspace(0, 1, n)[:, None]
        return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2

    def arc_to_points(p0, rx, ry, phi, large_arc, sweep, p1, n=None):
        n = n or resolution
        """椭圆弧转折线（端点参数化）。"""
        x0, y0 = p0
        x1, y1 = p1
        if rx == 0 or ry == 0 or (x0 == x1 and y0 == y1):
            return np.asarray([p0, p1], dtype=np.float64)
        rx, ry = abs(rx), abs(ry)
        phi_rad = math.radians(phi)
        cos_phi, sin_phi = math.cos(phi_rad), math.sin(phi_rad)
        dx2, dy2 = (x0 - x1) / 2.0, (y0 - y1) / 2.0
        x1p = cos_phi * dx2 + sin_phi * dy2
        y1p = -sin_phi * dx2 + cos_phi * dy2
        lam = (x1p ** 2) / (rx ** 2) + (y1p ** 2) / (ry ** 2)
        if lam > 1:
            rx *= math.sqrt(lam)
            ry *= math.sqrt(lam)
        sign = -1 if large_arc == sweep else 1
        num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
        den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
        co = sign * math.sqrt(max(0.0, num / den)) if den else 0.0
        cxp = co * rx * y1p / ry
        cyp = -co * ry * x1p / rx
        cx = cos_phi * cxp - sin_phi * cyp + (x0 + x1) / 2.0
        cy = sin_phi * cxp + cos_phi * cyp + (y0 + y1) / 2.0
        theta1 = math.atan2((y1p - cyp) / ry, (x1p - cxp) / rx)
        theta2 = math.atan2((-y1p - cyp) / ry, (-x1p - cxp) / rx)
        dtheta = theta2 - theta1
        if not sweep and dtheta > 0:
            dtheta -= 2 * math.pi
        elif sweep and dtheta < 0:
            dtheta += 2 * math.pi
        t = np.linspace(theta1, theta1 + dtheta, n)[:, None]
        xs = cos_phi * rx * np.cos(t) - sin_phi * ry * np.sin(t) + cx
        ys = sin_phi * rx * np.cos(t) + cos_phi * ry * np.sin(t) + cy
        return np.hstack([xs, ys])

    while i < len(tokens):
        kind, value = tokens[i]
        if kind == "cmd":
            cmd = value
            i += 1
            nums = []
            if cmd in "Zz":
                if pts:
                    pts.append(start)
                    flush()
                cur = start
                continue
        else:
            if not cmd:
                i += 1
                continue

        relative = cmd.islower()
        c = cmd.upper()

        def take(n: int) -> list[float] | None:
            nonlocal i
            vals = []
            for _ in range(n):
                if i >= len(tokens) or tokens[i][0] != "num":
                    return None
                vals.append(float(tokens[i][1]))
                i += 1
            return vals

        if c == "M":
            v = take(2)
            if v is None:
                break
            x, y = v
            if relative:
                x += cur[0]
                y += cur[1]
            flush()
            pts = [(x, y)]
            cur = (x, y)
            start = (x, y)
            cmd = "l" if relative else "L"
        elif c == "L":
            v = take(2)
            if v is None:
                break
            x, y = v
            if relative:
                x += cur[0]
                y += cur[1]
            pts.append((x, y))
            cur = (x, y)
        elif c == "H":
            v = take(1)
            if v is None:
                break
            x = v[0] + (cur[0] if relative else 0.0)
            pts.append((x, cur[1]))
            cur = (x, cur[1])
        elif c == "V":
            v = take(1)
            if v is None:
                break
            y = v[0] + (cur[1] if relative else 0.0)
            pts.append((cur[0], y))
            cur = (cur[0], y)
        elif c == "C":
            v = take(6)
            if v is None:
                break
            p0 = np.asarray(cur, dtype=np.float64)
            c1 = np.asarray(v[0:2], dtype=np.float64) + (p0 if relative else 0)
            c2 = np.asarray(v[2:4], dtype=np.float64) + (p0 if relative else 0)
            p3 = np.asarray(v[4:6], dtype=np.float64) + (p0 if relative else 0)
            seg = bezier(p0, c1, c2, p3)
            pts.extend(map(tuple, seg[1:]))
            cur = tuple(p3)
        elif c == "S":
            v = take(4)
            if v is None:
                break
            p0 = np.asarray(cur, dtype=np.float64)
            p3 = np.asarray(v[2:4], dtype=np.float64) + (p0 if relative else 0)
            prev = np.asarray(pts[-2], dtype=np.float64) if len(pts) >= 2 else p0
            c1 = 2 * p0 - prev
            c2 = np.asarray(v[0:2], dtype=np.float64) + (p0 if relative else 0)
            seg = bezier(p0, c1, c2, p3)
            pts.extend(map(tuple, seg[1:]))
            cur = tuple(p3)
        elif c == "Q":
            v = take(4)
            if v is None:
                break
            p0 = np.asarray(cur, dtype=np.float64)
            c1 = np.asarray(v[0:2], dtype=np.float64) + (p0 if relative else 0)
            p3 = np.asarray(v[2:4], dtype=np.float64) + (p0 if relative else 0)
            seg = quad(p0, c1, p3)
            pts.extend(map(tuple, seg[1:]))
            cur = tuple(p3)
        elif c == "T":
            v = take(2)
            if v is None:
                break
            p0 = np.asarray(cur, dtype=np.float64)
            p3 = np.asarray(v[0:2], dtype=np.float64) + (p0 if relative else 0)
            prev = np.asarray(pts[-2], dtype=np.float64) if len(pts) >= 2 else p0
            c1 = 2 * p0 - prev
            seg = quad(p0, c1, p3)
            pts.extend(map(tuple, seg[1:]))
            cur = tuple(p3)
        elif c == "A":
            v = take(7)
            if v is None:
                break
            p0 = np.asarray(cur, dtype=np.float64)
            p1 = np.asarray(v[5:7], dtype=np.float64) + (p0 if relative else 0)
            seg = arc_to_points(tuple(p0), v[0], v[1], v[2], int(v[3]), int(v[4]), tuple(p1))
            pts.extend(map(tuple, seg[1:]))
            cur = tuple(p1)
        else:
            i += 1

    flush()
    if offset != (0.0, 0.0):
        dx, dy = offset
        subpaths = [p + np.array([dx, dy]) for p in subpaths]
    return subpaths


def flatten_svg_paths(d_list: list[str], resolution: int = 24) -> list[tuple[np.ndarray, int]]:
    """展平一批 d 字符串，返回 [(polyline, path_index)]。"""
    out = []
    for idx, d in enumerate(d_list):
        for poly in parse_path(d, resolution=resolution):
            out.append((poly, idx))
    return out


def fill_even_odd(shape: tuple[int, int], polygons: list[np.ndarray]) -> np.ndarray:
    """按 even-odd 规则填充多边形集合，返回 bool 掩膜（按包围盒裁剪，避免全图开销）。"""
    mask = np.zeros(shape, dtype=bool)
    xor_cropped(mask, polygons)
    return mask


def xor_cropped(mask: np.ndarray, polygons: list[np.ndarray],
                value: bool = True) -> np.ndarray:
    """把多个多边形按 even-odd 异或进掩膜，只处理每个多边形的包围盒区域。"""
    h, w = mask.shape
    for poly in polygons:
        if len(poly) < 3:
            continue
        x0 = int(np.floor(poly[:, 0].min()))
        y0 = int(np.floor(poly[:, 1].min()))
        x1 = int(np.ceil(poly[:, 0].max())) + 1
        y1 = int(np.ceil(poly[:, 1].max())) + 1
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        sub = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        pts = np.round(poly - np.array([x0, y0])).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(sub, [pts], 1)
        region = mask[y0:y1, x0:x1]
        if value:
            np.logical_xor(region, sub.astype(bool), out=region)
        else:
            np.logical_or(region, sub.astype(bool), out=region)
    return mask


def _grad_id(fill_value: str) -> str:
    """从 url(#id) 中取出渐变 id。"""
    v = fill_value.strip()
    if v.startswith("url("):
        v = v[4:].rstrip(")").lstrip("#")
    return v


def parse_svg_layers(text: str, transform_of=None) -> tuple[dict, list[dict]]:
    """解析 SVG 的渐变定义与图层，支持嵌套分组（按字分组的输出结构）。

    规则：带 fill 属性的 <g> 是一个颜色层；外层分组（如 glyph-N）不带
    fill，其路径归入最近的带 fill 祖先；层的吐出顺序 = 文档书写顺序。
    transform_of(attrs) 由调用方提供，用于把 transform 转成平移量。
    """
    if transform_of is None:
        transform_of = lambda attrs: (0.0, 0.0)

    grads: dict[str, dict] = {}
    for gm in re.finditer(r"<(linearGradient|radialGradient)([^>]*)>(.*?)</\1>", text, re.S):
        attrs = gm.group(2)
        gid_m = re.search(r'id="([^"]+)"', attrs)
        if not gid_m:
            continue
        stops = [(float(o), c) for o, c in re.findall(
            r'<stop offset="([\d.]+)" stop-color="([^"]+)"', gm.group(3))]

        def attr(name: str, default: float = 0.0) -> float:
            mm = re.search(rf'{name}="([-\d.]+)"', attrs)
            return float(mm.group(1)) if mm else default

        grads[gid_m.group(1)] = {
            "type": gm.group(1),
            "stops": stops,
            "x1": attr("x1"), "y1": attr("y1"),
            "x2": attr("x2", 1.0), "y2": attr("y2", 1.0),
            "cx": attr("cx"), "cy": attr("cy"), "r": attr("r", 1.0),
        }

    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    layers = []

    def visit(element, inherited_fill='#000000', inherited_opacity=1., offset=(0., 0.), stroke=None, stroke_width=1.):
        stroke=element.get('stroke',stroke)
        stroke_width=float(element.get('stroke-width',stroke_width))
        fill = element.get('fill', inherited_fill)
        opacity = inherited_opacity * float(element.get('opacity', '1'))
        local = transform_of(' ' + ' '.join(f'{key}="{value}"' for key,value in element.attrib.items()))
        current = (offset[0]+local[0], offset[1]+local[1])
        name = element.tag.rsplit('}',1)[-1]
        if name in {'defs','metadata','title','desc','clipPath','filter'}:
            return
        if name == 'path':
            d = element.get('d')
            if d and (fill != 'none' or stroke):
                layers.append({'fill':fill,'opacity':opacity * float(element.get('fill-opacity','1')),
                               'stroke':stroke,'stroke_width':stroke_width,'d':[(d,current)]})
            return
        pending = []
        def flush():
            if pending:
                layers.append({'fill':fill,'opacity':opacity,'d':list(pending),
                               'stroke':stroke,'stroke_width':stroke_width})
                pending.clear()
        for child in element:
            tag = child.tag.rsplit('}',1)[-1]
            if tag == 'path' and not any(key in child.attrib for key in ('fill','opacity','fill-opacity','stroke','stroke-width')):
                if (fill != 'none' or stroke) and child.get('d'):
                    local = transform_of(' ' + ' '.join(f'{key}="{value}"' for key,value in child.attrib.items()))
                    pending.append((child.get('d'), (current[0]+local[0],current[1]+local[1])))
            else:
                flush()
                visit(child,fill,opacity,current,stroke,stroke_width)
        flush()

    visit(root)
    for layer in layers:
        fill = layer['fill']
        layer['stops'] = grads.get(_grad_id(fill)) if fill.startswith('url(') else None
    return grads, layers


def translate_path(d: str, offset: tuple[float,float]) -> str:
    """Apply translation without replacing cubic controls with sampled polygons."""
    if offset == (0.,0.):
        return d
    tokens = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?',d)
    sizes = {'M':2,'L':2,'H':1,'V':1,'C':6,'S':4,'Q':4,'T':2,'A':7}
    result = []
    command = ''
    index = 0
    first = True
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            result.append(command)
            index += 1
            if command.upper() == 'Z':
                continue
        size = sizes[command.upper()]
        values = list(map(float,tokens[index:index+size]))
        if len(values)!=size:
            raise ValueError('SVG 路径参数不完整')
        if command.isupper() or (first and command=='m'):
            if command.upper()=='H':values[0]+=offset[0]
            elif command.upper()=='V':values[0]+=offset[1]
            elif command.upper()=='A':
                values[5]+=offset[0];values[6]+=offset[1]
            else:
                for j in range(size):values[j]+=offset[j%2]
        result.extend(f'{value:.6f}'.rstrip('0').rstrip('.') if value else '0' for value in values)
        index += size
        first = False
        if command=='m':command='l'
        elif command=='M':command='L'
    return ' '.join(result)


class SvgRaster:
    """把生成的 SVG 栅格化，用于输出复核与指标计算。"""

    def __init__(self, svg_text: str):
        self.svg = svg_text
        self.layers: list[dict] = []
        self._parse()

    @staticmethod
    def _transform_offset(attrs: str) -> tuple[float, float]:
        m = re.search(r'transform="translate\(\s*([-\d.]+)\s*[ ,]?\s*([-\d.]*)\s*\)"', attrs)
        if not m:
            return (0.0, 0.0)
        dx = float(m.group(1))
        dy = float(m.group(2)) if m.group(2) else 0.0
        return (dx, dy)

    def _parse(self) -> None:
        text = self.svg
        w = re.search(r'width="(\d+)"', text)
        h = re.search(r'height="(\d+)"', text)
        self.width = int(w.group(1)) if w else 0
        self.height = int(h.group(1)) if h else 0
        _, self.layers = parse_svg_layers(text, transform_of=self._transform_offset)

    def rasterize(self, background: tuple[int, int, int, int] = (0, 0, 0, 0),
                  scale: int = 2) -> np.ndarray:
        """Return straight RGBA pixels from resvg, including SVG filters."""
        from .native_render import render_svg
        return render_svg(self.svg, background)

    def rasterize_legacy(self, background: tuple[int, int, int, int] = (0, 0, 0, 0),
                         scale: int = 2) -> np.ndarray:
        """按浏览器语义栅格化：path 元素内部 even-odd 挖孔，path 之间相互叠加。

        scale>1 时在超采样分辨率下求覆盖掩膜再降采样，使边界覆盖率接近真实
        渲染器的抗锯齿结果，保证指标口径与浏览器看到的效果一致。
        """
        w, h = self.width, self.height
        sw, sh = w * max(1, scale), h * max(1, scale)
        canvas = np.zeros((h, w, 4), dtype=np.float64)
        canvas[:, :] = background
        scratch = np.empty((sh, sw), dtype=bool)

        for layer in self.layers:
            a = layer["opacity"]
            grad = layer["stops"]
            color_img = self._gradient_canvas(grad, h, w) if (
                layer["fill"].startswith("url(") and grad) else None
            color = None if color_img is not None else np.asarray(
                _hex_to_rgb(layer.get('stroke') or layer['fill']), dtype=np.float64)

            layer_mask = np.zeros((sh, sw), dtype=bool)
            painted = False
            for item in layer["d"]:
                if isinstance(item, tuple):
                    d, offset = item
                else:
                    d, offset = item, (0.0, 0.0)
                polys = parse_path(d, offset, resolution=12)
                if not polys:
                    continue
                scratch.fill(False)
                if layer.get('stroke'):
                    marks=np.zeros((sh,sw),np.uint8)
                    thickness=max(1,int(round(layer['stroke_width']*scale)))
                    for poly in polys:
                        points=np.round(poly*scale).astype(np.int32)
                        cv2.polylines(marks,[points],False,1,thickness,cv2.LINE_8)
                        for point in (points[0],points[-1]):
                            cv2.circle(marks,tuple(point),max(1,thickness//2),1,-1)
                    scratch[:]=marks>0
                else:
                    xor_cropped(scratch, [p * scale for p in polys])
                if scratch.any():
                    np.logical_or(layer_mask, scratch, out=layer_mask)
                    painted = True
            if not painted:
                continue

            if scale > 1:
                coverage = cv2.resize((layer_mask.astype(np.uint8) * 255), (w, h),
                                      interpolation=cv2.INTER_AREA) / 255.0
            else:
                coverage = layer_mask.astype(np.float64)
            mask = coverage >= 0.5
            if not mask.any():
                continue

            src_color = color_img[mask] if color_img is not None else color
            eff = (a * coverage[mask])[:, None]
            canvas[mask, 0:3] = canvas[mask, 0:3] * (1 - eff) + src_color * eff
            canvas[mask, 3] = canvas[mask, 3] * (1 - a * coverage[mask]) + 255.0 * a * coverage[mask]

        return np.clip(canvas, 0, 255).astype(np.uint8)

    @staticmethod
    def _gradient_canvas(grad: dict, h: int, w: int, scale: int = 1) -> np.ndarray:
        """按使用者坐标系还原线性/径向渐变（贡献配色色阶），供指标计算用。"""
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        if "radial" in str(grad.get("type", "")):
            cx, cy = grad.get("cx", 0.0) * scale, grad.get("cy", 0.0) * scale
            r = max(1e-9, grad.get("r", 1.0) * scale)
            t = np.clip(np.hypot(xx - cx, yy - cy) / r, 0.0, 1.0)
        else:
            x1, y1 = grad["x1"] * scale, grad["y1"] * scale
            x2, y2 = grad["x2"] * scale, grad["y2"] * scale
            dx, dy = x2 - x1, y2 - y1
            denom = dx * dx + dy * dy
            if denom < 1e-9:
                t = np.zeros((h, w), dtype=np.float64)
            else:
                t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / denom, 0.0, 1.0)
        stops = sorted(grad["stops"], key=lambda s: s[0])
        offsets = np.asarray([s[0] for s in stops], dtype=np.float64)
        cols = np.asarray([_hex_to_rgb(c) for _, c in stops], dtype=np.float64)
        out = np.zeros((h, w, 3), dtype=np.float64)
        for ch in range(3):
            out[:, :, ch] = np.interp(t, offsets, cols[:, ch])
        return out

    def node_stats(self) -> dict:
        from .svgutil import count_nodes_from_d
        per_path = []
        subpaths = 0
        for layer in self.layers:
            for item in layer["d"]:
                d = item[0] if isinstance(item, tuple) else item
                n = count_nodes_from_d(d)
                if n:
                    per_path.append(n)
                subpaths += d.count("M") + d.count("m")
        return {
            "path_count": len(per_path),
            "subpath_count": int(subpaths),
            "node_total": int(sum(per_path)),
            "node_max": int(max(per_path)) if per_path else 0,
            "node_min": int(min(per_path)) if per_path else 0,
            "node_mean": float(np.mean(per_path)) if per_path else 0.0,
            "layer_count": len(self.layers),
        }


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(ch * 2 for ch in v)
    if len(v) < 6:
        return (0, 0, 0)
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
