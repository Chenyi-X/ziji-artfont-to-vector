# -*- coding: utf-8 -*-
"""两段式筛查：预处理质量判定先剔除场景照，仅对通过者做完整转换。"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.pipeline.preprocess import preprocess        # noqa: E402
from app.pipeline.service import convert_bytes        # noqa: E402

TMP = Path(__file__).parent
CAND = TMP / "candidates"


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                done[d["file"]] = d
            except Exception:
                pass
    return done


def main():
    a, b = int(sys.argv[1]), int(sys.argv[2])
    manifest_path = Path(sys.argv[3]) if len(sys.argv) > 3 else TMP / "candidates.json"
    cand_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else CAND
    out_path = Path(sys.argv[5]) if len(sys.argv) > 5 else TMP / "screen2.jsonl"
    stage_path = out_path
    done = load_done(stage_path)
    fout = open(stage_path, "a", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for i, m in enumerate(manifest, 1):
        if not (a <= i <= b) or m["file"] in done:
            continue
        p = cand_dir / m["file"]
        if not p.exists():
            continue
        data = p.read_bytes()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            rec = {**m, "error": "无法解码"}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"[{i:3d}] {m['file']} 无法解码", flush=True)
            continue
        t0 = time.perf_counter()
        prep = preprocess(img)
        qd = prep.quality.to_dict()
        pre_ms = (time.perf_counter() - t0) * 1000
        base = {**m, "invalid": qd.get("invalid"), "reasons": qd.get("invalid_reasons"),
                "photo_like": qd.get("photo_like"), "bbox_ratio": qd.get("orig_bbox_ratio"),
                "severity": qd.get("severity"), "pre_ms": round(pre_ms)}
        if qd.get("invalid") or qd.get("photo_like") or qd.get("severity") == "reject":
            rec = {**base, "stage": "rejected_by_quality"}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"[{i:3d}] {m['file']:16s} 剔除 reasons={qd.get('invalid_reasons')} "
                  f"bbox={qd.get('orig_bbox_ratio')} pre={pre_ms:.0f}ms", flush=True)
            continue
        try:
            r = convert_bytes(m["file"], data, with_preview=False, strict_input=False)
        except Exception as e:
            rec = {**base, "error": f"异常: {e}", "stage": "convert_error"}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"[{i:3d}] {m['file']} 转换异常: {e}", flush=True)
            continue
        if r.error:
            rec = {**base, "error": r.error, "stage": "convert_error"}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"[{i:3d}] {m['file']} 转换失败: {r.error[:60]}", flush=True)
            continue
        mt = r.metrics
        rec = {**base, "stage": "converted",
               "contour": mt.get("contour_fidelity"),
               "feature": mt.get("feature_retention"),
               "node_pg": mt.get("node_per_glyph"),
               "color_dev": mt.get("color_deviation"),
               "color_mode": mt.get("color_mode"),
               "layers": mt.get("layer_count"),
               "elapsed_ms": r.elapsed_ms,
               "checks_fail": [k for k, v in mt.get("checks", {}).items() if not v]}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        print(f"[{i:3d}] {m['file']:16s} 可用 bbox={qd.get('orig_bbox_ratio')}"
              f" contour={mt.get('contour_fidelity'):.2f} node={mt.get('node_per_glyph'):.0f}"
              f" {r.elapsed_ms:.0f}ms", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
