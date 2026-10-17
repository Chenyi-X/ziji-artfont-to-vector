"""浏览器渲染一致性复核：用 Chrome / Edge 无头模式直接渲染 SVG，与内部栅格化对比。

需求三节「输出 SVG 在 Chrome/Firefox/Edge/Safari 中渲染一致」的实测部分：
本机装了什么浏览器就实测什么；Safari 需要在 macOS 上由用户补充验证，
Firefox 未安装时同样标注未验证。

用法：python eval/browser_render_check.py -o out/browser_render
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.pipeline.pathparse import SvgRaster  # noqa: E402
from app.pipeline.service import convert_bytes  # noqa: E402
from app.pipeline.vectorize import VectorizeOptions  # noqa: E402

BROWSERS = {
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
}


def render_with(exe: str, html_path: str, out_png: Path, w: int, h: int) -> bool:
    cmd = [exe, "--headless", "--disable-gpu", "--hide-scrollbars",
           f"--window-size={w},{h}", f"--screenshot={out_png}", f"file:///{html_path}"]
    if "firefox" in exe.lower():
        cmd = [exe, "--headless", "--screenshot", str(out_png), f"file:///{html_path}"]
    try:
        subprocess.run(cmd, timeout=60, capture_output=True)
    except Exception:
        return False
    return out_png.exists()


BG = (128, 128, 128)


def fg_mask(img: np.ndarray, bg=BG, thr: float = 60.0) -> np.ndarray:
    """前景掩膜：与页面底色的色差 >thr 视为前景（截图无 alpha）。

    用中灰底而不是白底：白色艺术字/白色装饰点在白底截图里不可见，
    会把渲染一致度测低。
    """
    bgr = img[:, :, :3].astype(np.float64)
    dist = np.linalg.norm(bgr - np.asarray(bg, dtype=np.float64), axis=2)
    return dist > thr


def internal_mask(scale1_rgba: np.ndarray, bg=BG, thr: float = 60.0) -> np.ndarray:
    """内部栅格在同底色上的合成掩膜。"""
    rgba = scale1_rgba.astype(np.float64)
    a = rgba[:, :, 3:4] / 255.0
    comp = rgba[:, :, :3] * a + np.asarray(bg, dtype=np.float64) * (1 - a)
    dist = np.linalg.norm(comp - np.asarray(bg, dtype=np.float64), axis=2)
    return dist > thr


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def dilate(mask: np.ndarray, px: int = 1) -> np.ndarray:
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)


def boundary_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """对称边界距离：平均表面距离（px）与 95 分位（px）。

    浏览器与内部栅格化的抗锯齿伽马不同，同一边界点的二值化结果会差 1~2px，
    掩膜 IoU 会被这个系统性偏差拖低；边界距离直接量几何位置是否一致。
    """
    ba, bb = boundary(a), boundary(b)
    if not ba.any() or not bb.any():
        return 99.0, 99.0
    da = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 3)
    d1, d2 = db[ba], da[bb]
    all_d = np.concatenate([d1, d2])
    return float(all_d.mean()), float(np.percentile(all_d, 95))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", default=str(ROOT / "out" / "browser_render"))
    parser.add_argument("--samples", type=int, default=0, help="0 = 全部样本")
    args = parser.parse_args(argv)

    outdir = Path(args.out)
    tmp = ROOT / "out" / "browser_check_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    mf = ROOT / "samples" / "manifest.json"
    if mf.exists():
        for item in json.loads(mf.read_text(encoding="utf-8")):
            manifest[item["file"]] = item

    installed = {k: v for k, v in BROWSERS.items() if Path(v).exists()}
    if not installed:
        print("未找到任何浏览器")
        return 1

    files = sorted((ROOT / "samples" / "images").glob("*.png"))
    if args.samples:
        files = files[: args.samples]

    opts = VectorizeOptions.from_preset("balanced")
    per_browser: dict[str, list[dict]] = {k: [] for k in installed}
    for path in files:
        item = manifest.get(path.name, {})
        g = item.get("glyphs")
        res = convert_bytes(path.name, path.read_bytes(), opts, with_preview=False,
                            strict_input=False, glyph_count=g, glyph_text=item.get("text"))
        if res.error:
            print(f"  {path.name} 转换失败：{res.error}")
            continue
        svg_path = tmp / f"{path.stem}.svg"
        svg_path.write_text(res.svg, encoding="utf-8")
        w, h = res.size["width"], res.size["height"]
        # 灰底页面承载 SVG，白色元素也能被截图对比到
        html_path = tmp / f"{path.stem}.html"
        html_path.write_text(
            f'<body style="margin:0;background:rgb{BG}">'
            f'<img src="{path.stem}.svg" width="{w}" height="{h}"></body>',
            encoding="utf-8")
        internal = SvgRaster(res.svg).rasterize(scale=1)
        int_fg = internal_mask(internal)
        for name, exe in installed.items():
            png = tmp / f"{path.stem}_{name}.png"
            ok = render_with(exe, str(html_path).replace("\\", "/"), png, w, h)
            if not ok:
                per_browser[name].append({"file": path.name, "error": "渲染失败"})
                continue
            # cv2.imread 不支持非 ASCII 路径，用 imdecode 读
            try:
                buf = np.fromfile(png, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
            except OSError:
                img = None
            if img is None or img.shape[0] < h - 2:
                per_browser[name].append({"file": path.name, "error": "截图尺寸不符"})
                continue
            img = img[:h, :w]
            bmask = fg_mask(img)
            bmean, bp95 = boundary_stats(int_fg, bmask)
            per_browser[name].append({"file": path.name,
                                      "iou": round(iou(int_fg, bmask), 4),
                                      "boundary_mean_px": round(bmean, 3),
                                      "boundary_p95_px": round(bp95, 3)})
        print(f"  {path.name} 完成")

    summary = {}
    for name, items in per_browser.items():
        ok = [x["iou"] for x in items if "iou" in x]
        bm = [x["boundary_mean_px"] for x in items if "boundary_mean_px" in x]
        bp = [x["boundary_p95_px"] for x in items if "boundary_p95_px" in x]
        summary[name] = {
            "count": len(ok),
            "avg_iou": round(float(np.mean(ok)), 4) if ok else None,
            "min_iou": round(float(np.min(ok)), 4) if ok else None,
            "avg_boundary_mean_px": round(float(np.mean(bm)), 3) if bm else None,
            "max_boundary_mean_px": round(float(np.max(bm)), 3) if bm else None,
            "max_boundary_p95_px": round(float(np.max(bp)), 3) if bp else None,
            "boundary_over_1px": [x["file"] for x in items
                                   if x.get("boundary_mean_px", 0) > 1.0],
            "errors": [x for x in items if "error" in x],
        }
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "browsers": summary,
              "items": per_browser,
              "untested": ["firefox（未安装）", "safari（需 macOS 环境，待用户验证）",
                           "Illustrator 2020+（待用户验证）"]}
    (outdir / "browser_render.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n==== 浏览器渲染一致性（vs 内部栅格化） ====")
    for name, s in summary.items():
        if s["avg_iou"] is None:
            print(f"{name}: 无有效结果")
        else:
            print(f"{name}: 掩膜 IoU 平均 {s['avg_iou'] * 100:.2f}%（受抗锯齿阈值口径影响）；"
                  f"边界平均偏差 {s['avg_boundary_mean_px']}px（最差样本 {s['max_boundary_mean_px']}px，"
                  f"P95 最大 {s['max_boundary_p95_px']}px，>1px 的样本 {len(s['boundary_over_1px'])} 个）")
    print("未实测：", "；".join(report["untested"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
