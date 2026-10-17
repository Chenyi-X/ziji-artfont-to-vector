# -*- coding: utf-8 -*-
"""第二轮采集：平面印刷/图形类艺术字（扫描件、徽章、标签、粉笔字等）。"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

TMP = Path(__file__).parent
CAND = TMP / "candidates2"
CAND.mkdir(exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) material-research"}

QUERIES = [
    ("woodtype", "wood type letterpress specimen"),
    ("specimen", "type specimen book letter"),
    ("vintage_ad", "vintage advertisement label typography"),
    ("artnouveau", "art nouveau poster lettering"),
    ("badge", "badge emblem lettering crest"),
    ("chalk", "chalkboard menu lettering"),
    ("bookcover", "book cover lettering design"),
    ("packaging", "packaging label typography"),
    ("ornament", "ornamental decorated letter"),
    ("headline", "newspaper headline masthead letter"),
    ("seal", "chinese seal script print"),
    ("embroidery", "embroidery letter patch"),
]

API = "https://api.openverse.org/v1/images/"


def search(q, n=20):
    params = urllib.parse.urlencode({
        "q": q, "license": "cc0,pdm,by", "page_size": n, "mature": "false"})
    req = urllib.request.Request(API + "?" + params, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    old = json.loads((TMP / "candidates.json").read_text(encoding="utf-8"))
    seen = {m["url"] for m in old}
    manifest = []
    idx = 200
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
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=40) as resp:
                    data = resp.read()
            except Exception as e:
                print(f"[下载失败] {url[:70]}: {e}")
                continue
            if len(data) > 5 * 1024 * 1024:
                continue
            (CAND / name).write_bytes(data)
            manifest.append({
                "file": name, "style": style, "query": q,
                "title": (r.get("title") or "")[:120],
                "creator": (r.get("creator") or "")[:80],
                "license": f"{r.get('license')} {r.get('license_version') or ''}".strip(),
                "source_page": r.get("foreign_landing_url") or "",
                "url": url, "bytes": len(data), "width": w, "height": h,
            })
            got += 1
            print(f"[OK] {name} {w}x{h} {len(data)//1024}KB | {r.get('title','')[:40]}")
            if got >= 8:
                break
        time.sleep(1.0)
    (TMP / "candidates2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n第二轮下载 {len(manifest)} 张")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
