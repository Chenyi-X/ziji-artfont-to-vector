"""浏览器一致性校验：用真实 SVG 渲染器（resvg）复核内部栅格化与指标口径。

用法：
  python eval/browser_check.py -o out/samples            # 校验已有 SVG
  python eval/browser_check.py -o out/samples --convert   # 重新转换后校验（可加 --samples N）

校验内容：
  1. 内部栅格化 vs resvg 渲染的轮廓 IoU（两者应当几乎完全一致）
  2. 以 resvg 渲染为准的轮廓还原度（真实浏览器/Illustrator 看到的效果）
  3. 内孔是否被正确挖空（对比原图前景掩膜里的“孔”的数量与位置）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.pipeline.pathparse import SvgRaster  # noqa: E402
from app.pipeline.preprocess import preprocess  # noqa: E402
from app.pipeline.service import convert_bytes  # noqa: E402
from app.pipeline.vectorize import VectorizeOptions  # noqa: E402

NODE = "node"
RENDERER = ROOT / "eval" / "tools" / "render_svg.mjs"
SAMPLES = ROOT / "samples" / "images"


def render_with_resvg(svg_path: Path, png_path: Path, width: int | None = None) -> tuple[int, int]:
    cmd = [NODE, str(RENDERER), str(Path(svg_path).resolve()), str(Path(png_path).resolve())]
    if width:
        cmd.append(str(width))
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(RENDERER.parent))
    if out.returncode != 0:
        raise RuntimeError(f"resvg 渲染失败：{(out.stderr or '').strip()}")
    info = json.loads(out.stdout.strip().splitlines()[-1])
    return int(info["width"]), int(info["height"])


def imread_unicode(path: Path):
    """cv2.imread 在 Windows 下不认中文路径，用字节流解码。"""
    return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def count_holes(mask: np.ndarray) -> int:
    """统计前景中的孔洞数量（前景外部轮廓数 - 前景团块数）。"""
    m = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int(sum(1 for h in hierarchy[0] if h[3] != -1))


def check_one(svg_path: Path, image_path: Path, tmp: Path) -> dict:
    svg_text = svg_path.read_text(encoding="utf-8")
    internal = SvgRaster(svg_text).rasterize()
    internal_mask = internal[:, :, 3] > 127

    png = tmp / (svg_path.stem + ".png")
    w, h = render_with_resvg(svg_path, png)
    shot = imread_unicode(png)
    if shot is None:
        return {"svg": svg_path.name, "error": "渲染结果读取失败"}
    if shot.shape[2] == 4:
        resvg_mask = shot[:, :, 3] > 127
    else:
        gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        resvg_mask = gray < 250

    if resvg_mask.shape != internal_mask.shape:
        resvg_mask = cv2.resize(resvg_mask.astype(np.uint8), (internal_mask.shape[1], internal_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0

    match = iou(internal_mask, resvg_mask)
    result = {
        "svg": svg_path.name,
        "internal_vs_resvg_iou": round(match, 4),
        "resvg_pixels": int(resvg_mask.sum()),
        "internal_pixels": int(internal_mask.sum()),
    }
    if image_path.exists():
        prepared = preprocess(imread_unicode(image_path))
        orig = prepared.mask > 127
        result["fidelity_internal"] = round(iou(orig, internal_mask), 4)
        result["fidelity_resvg"] = round(iou(orig, resvg_mask), 4)
        result["holes_orig"] = count_holes(orig)
        result["holes_resvg"] = count_holes(resvg_mask)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--svg-dir", default=str(ROOT / "out" / "samples"))
    parser.add_argument("--img-dir", default=str(SAMPLES))
    parser.add_argument("--convert", action="store_true", help="先用当前代码重新转换样本")
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", default=str(ROOT / "out" / "browser_check.json"))
    args = parser.parse_args(argv)

    svg_dir = Path(args.svg_dir)
    img_dir = Path(args.img_dir)

    if args.convert:
        opts = VectorizeOptions.from_preset(args.preset)
        for img in sorted(img_dir.glob("*.png")):
            res = convert_bytes(img.name, img.read_bytes(), opts, with_preview=False, strict_input=False)
            if not res.error:
                (svg_dir / f"{img.stem}.svg").write_text(res.svg, encoding="utf-8")

    files = sorted(svg_dir.glob("*.svg"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("没有可校验的 SVG")
        return 1

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for svg in files:
            row = check_one(svg, img_dir / f"{svg.stem}.png", tmp)
            rows.append(row)
            if "error" in row:
                print(f"  {row['svg']:26s} {row['error']}")
                continue
            extra = ""
            if "fidelity_resvg" in row:
                extra = (f"  真实渲染还原度 {row['fidelity_resvg'] * 100:5.1f}%"
                         f"  （内部 {row['fidelity_internal'] * 100:5.1f}%）"
                         f"  孔洞 {row['holes_resvg']}/{row['holes_orig']}")
            print(f"  {row['svg']:26s} 内部 vs resvg 一致度 {row['internal_vs_resvg_iou'] * 100:6.2f}%{extra}")

    ok = [r for r in rows if "error" not in r]
    matches = [r["internal_vs_resvg_iou"] for r in ok]
    summary = {
        "count": len(rows),
        "min_internal_vs_resvg_iou": round(min(matches), 4) if matches else 0,
        "avg_internal_vs_resvg_iou": round(float(np.mean(matches)), 4) if matches else 0,
        "min_resvg_fidelity": round(min((r.get("fidelity_resvg", 0) for r in ok), default=0), 4),
        "avg_resvg_fidelity": round(float(np.mean([r.get("fidelity_resvg", 0) for r in ok]))
                                    if ok else 0, 4),
        "hole_mismatch": [r["svg"] for r in ok
                          if r.get("holes_orig") is not None and r["holes_resvg"] < r["holes_orig"]],
    }
    Path(args.report).write_text(json.dumps({"summary": summary, "items": rows},
                                            ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n==== 浏览器一致性汇总 ====")
    print(f"内部栅格化 vs 真实渲染 最小一致度 {summary['min_internal_vs_resvg_iou'] * 100:.2f}%"
          f"（平均 {summary['avg_internal_vs_resvg_iou'] * 100:.2f}%）")
    print(f"真实渲染口径平均还原度 {summary['avg_resvg_fidelity'] * 100:.1f}%"
          f"（最低 {summary['min_resvg_fidelity'] * 100:.1f}%）")
    print(f"孔洞丢失样本：{summary['hole_mismatch'] or '无'}")
    print(f"报告：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
