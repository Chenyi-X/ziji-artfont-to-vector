# -*- coding: utf-8 -*-
"""从 Openverse 采集真实艺术字候选素材（仅 cc0/pdm/by 授权）。"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

TMP = Path(__file__).parent
CAND = TMP / "candidates"
CAND.mkdir(exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) material-research"}

QUERIES = [
    ("calligraphy", "chinese calligraphy brush"),
    ("calligraphy", "calligraphy ink character"),
    ("handwritten", "hand lettering chalk"),
    ("handwritten", "handwritten lettering ink"),
    ("cartoon", "cartoon lettering"),
    ("cartoon", "comic book lettering"),
    ("gradient", "gradient gold lettering"),
    ("gradient", "colorful gradient typography"),
    ("stroke", "sign painting lettering"),
    ("stroke", "hand painted shop sign letters"),
    ("neon", "neon sign letters"),
    ("3d", "metal letters building sign"),
    ("serif", "letterpress type specimen poster"),
    ("logo", "logo typography wordmark"),
    ("graffiti", "graffiti letters"),
]

API = "https://api.openverse.org/v1/images/"


def search(q, n=20):
    params = urllib.parse.urlencode({
        "q": q, "license": "cc0,pdm,by", "page_size": n,
        "mature": "false",
    })
    req = urllib.request.Request(API + "?" + params, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    manifest = []
    seen = set()
    idx = 0
    for style, q in QUERIES:
        try:
            d = search(q)
        except Exception as e:
            print(f"[搜索失败] {q}: {e}")
            continue
        got = 0
        for r in d.get("results", []):
            url = r.get("url") or ""
            if not url or url in seen:
                continue
            w, h = r.get("width") or 0, r.get("height") or 0
            if min(w, h) and min(w, h) < 500:
                continue
            idx += 1
            seen.add(url)
            name = f"{idx:03d}_{style}.jpg"
            path = CAND / name
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=40) as resp:
                    data = resp.read()
            except Exception as e:
                print(f"[下载失败] {url[:70]}: {e}")
                continue
            if len(data) > 5 * 1024 * 1024:
                continue
            path.write_bytes(data)
            manifest.append({
                "file": name, "style": style, "query": q,
                "title": (r.get("title") or "")[:120],
                "creator": (r.get("creator") or "")[:80],
                "license": f"{r.get('license')} {r.get('license_version') or ''}".strip(),
                "source_page": r.get("foreign_landing_url") or "",
                "url": url, "bytes": len(data),
                "width": w, "height": h,
            })
            got += 1
            print(f"[OK] {name} {w}x{h} {len(data)//1024}KB | {r.get('title','')[:40]}")
            if got >= 8:
                break
        time.sleep(1.0)
    (TMP / "candidates.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n共下载 {len(manifest)} 张候选")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
