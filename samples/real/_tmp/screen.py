# -*- coding: utf-8 -*-
"""用项目流水线筛查候选素材：质量判定 + 转换指标。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.pipeline.service import convert_bytes  # noqa: E402

TMP = Path(__file__).parent
CAND = TMP / "candidates"


def main():
    a, b = int(sys.argv[1]), int(sys.argv[2])
    done_path = TMP / "screen.jsonl"
    done = set()
    if done_path.exists():
        for line in done_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                pass
    fout = open(done_path, "a", encoding="utf-8")
    manifest = json.loads((TMP / "candidates.json").read_text(encoding="utf-8"))
    out = []
    for i, m in enumerate(manifest, 1):
        if not (a <= i <= b) or m["file"] in done:
            continue
        p = CAND / m["file"]
        if not p.exists():
            continue
        data = p.read_bytes()
        try:
            r = convert_bytes(m["file"], data, with_preview=False, strict_input=False)
        except Exception as e:
            out.append({**m, "error": f"异常: {e}"})
            fout.write(json.dumps(out[-1], ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i:3d}] {m['file']} 异常: {e}", flush=True)
            continue
        if r.error:
            out.append({**m, "error": r.error})
            fout.write(json.dumps(out[-1], ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i:3d}] {m['file']} 失败: {r.error}")
            continue
        q, mt = r.quality, r.metrics
        out.append({
            **m,
            "invalid": q.get("invalid"),
            "reasons": q.get("invalid_reasons"),
            "photo_like": q.get("photo_like"),
            "bbox_ratio": q.get("orig_bbox_ratio"),
            "contour": mt.get("contour_fidelity"),
            "feature": mt.get("feature_retention"),
            "node_pg": mt.get("node_per_glyph"),
            "color_dev": mt.get("color_deviation"),
            "color_mode": mt.get("color_mode"),
            "layers": mt.get("layer_count"),
            "elapsed_ms": r.elapsed_ms,
            "checks_fail": [k for k, v in mt.get("checks", {}).items() if not v],
        })
        fout.write(json.dumps(out[-1], ensure_ascii=False) + "\n")
        fout.flush()
        flag = "剔除" if (q.get("invalid") or q.get("photo_like")) else "可用"
        print(f"[{i:3d}] {m['file']:16s} {flag} bbox={q.get('orig_bbox_ratio')}"
              f" contour={mt.get('contour_fidelity'):.2f} node={mt.get('node_per_glyph'):.0f}"
              f" {r.elapsed_ms:.0f}ms", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
