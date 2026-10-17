# -*- coding: utf-8 -*-
"""真实素材验收：samples/real 全量转换，产出 SVG、对比图与指标报告。

用法：
    python eval/run_real_materials.py --out out/real_materials --range 0 8
    python eval/run_real_materials.py --out out/real_materials --finalize
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline.preview import build_comparison, encode_png  # noqa: E402
from app.pipeline.service import convert_bytes                  # noqa: E402

SRC = ROOT / "samples" / "real"


def run_range(out: Path, a: int, b: int) -> None:
    files = sorted(p for p in SRC.glob("R*.jpg"))
    files = files[a:b]
    log = out / "items.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                pass
    out.mkdir(parents=True, exist_ok=True)
    (out / "svg").mkdir(exist_ok=True)
    (out / "cmp").mkdir(exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        for p in files:
            if p.name in done:
                continue
            t0 = time.perf_counter()
            r = convert_bytes(p.name, p.read_bytes(), with_preview=True, strict_input=False)
            rec = {"file": p.name, "error": r.error}
            if not r.error:
                (out / "svg" / f"{p.stem}.svg").write_text(r.svg, encoding="utf-8")
                if r.preview_png:
                    (out / "cmp" / f"{p.stem}.png").write_bytes(r.preview_png)
                rec.update({
                    "metrics": r.metrics, "quality": r.quality,
                    "elapsed_ms": round(r.elapsed_ms, 1), "warnings": r.warnings,
                })
            rec["wall_s"] = round(time.perf_counter() - t0, 1)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if r.error:
                print(f"[{p.name}] 失败: {r.error}", flush=True)
            else:
                m = r.metrics
                print(f"[{p.name}] contour={m.get('contour_fidelity'):.2f} "
                      f"feature={m.get('feature_retention'):.2f} "
                      f"node={m.get('node_per_glyph'):.0f} "
                      f"color={m.get('color_deviation') * 100:.1f}% "
                      f"{r.elapsed_ms / 1000:.1f}s", flush=True)


def finalize(out: Path) -> None:
    log = out / "items.jsonl"
    items = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        items[d["file"]] = d
    rows = []
    for name in sorted(items):
        it = items[name]
        if it.get("error"):
            rows.append({"file": name, "passed": False, "error": it["error"]})
            continue
        m, q = it["metrics"], it["quality"]
        checks = m.get("checks", {})
        fails = [k for k, v in checks.items() if not v]
        rows.append({
            "file": name, "passed": checks.get("passed", False),
            "fails": fails,
            "contour": m.get("contour_fidelity"),
            "feature": m.get("feature_retention"),
            "node_pg": m.get("node_per_glyph"),
            "color_dev": m.get("color_deviation"),
            "color_mode": m.get("color_mode"),
            "holes_ok": m.get("holes_preserved"),
            "bbox_ratio": q.get("orig_bbox_ratio"),
            "invalid": q.get("invalid"), "photo_like": q.get("photo_like"),
            "elapsed_ms": it.get("elapsed_ms"),
        })
    ok = [r for r in rows if r.get("passed")]
    conv = [r for r in rows if not r.get("error")]
    summary = {
        "count": len(rows), "passed": len(ok),
        "avg_contour": round(sum(r["contour"] for r in conv) / max(len(conv), 1), 4),
        "min_contour": round(min((r["contour"] for r in conv), default=0), 4),
        "avg_feature": round(sum(r["feature"] for r in conv) / max(len(conv), 1), 4),
        "avg_node": round(sum(r["node_pg"] for r in conv) / max(len(conv), 1), 1),
        "avg_elapsed_s": round(sum(r["elapsed_ms"] for r in conv) / max(len(conv), 1) / 1000, 2),
        "fail_reasons": {},
    }
    for r in conv:
        for k in r.get("fails", []):
            if k != "passed":
                summary["fail_reasons"][k] = summary["fail_reasons"].get(k, 0) + 1
    (out / "real_report.json").write_text(
        json.dumps({"summary": summary, "items": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    lines = ["# 真实素材验收报告", "",
             f"素材数 {summary['count']}，全项达标 {summary['passed']}。", "",
             "| 文件 | 达标 | 轮廓 | 特征 | 节点/字 | 色偏 | 耗时s | 未过项 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['file']} | - | - | - | - | - | - | {r['error'][:40]} |")
        else:
            lines.append(
                f"| {r['file']} | {'✅' if r['passed'] else '❌'} | {r['contour']:.2f} "
                f"| {r['feature']:.2f} | {r['node_pg']:.0f} | {r['color_dev'] * 100:.1f}% "
                f"| {r['elapsed_ms'] / 1000:.1f} | {','.join(r['fails'])} |")
    (out / "real_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/real_materials")
    ap.add_argument("--range", nargs=2, type=int, default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    if args.finalize:
        finalize(out)
    elif args.range:
        run_range(out, args.range[0], args.range[1])
    else:
        run_range(out, 0, 999)
