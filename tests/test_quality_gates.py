"""本轮新增口径的单元测试：直线/正交吸附、直线平直度、阴影分层。

运行：python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.metrics import orthogonal_deviation  # noqa: E402
from app.pipeline.preprocess import preprocess  # noqa: E402
from app.pipeline.svgutil import fit_contour, snap_orthogonal  # noqa: E402
from app.pipeline.vectorize import VectorizeOptions, vectorize  # noqa: E402


def _wavy_ring(n: int = 240, amp: float = 0.9) -> np.ndarray:
    """一条近水平的波浪长条轮廓（模拟抗锯齿抖动 + 拟合波浪）。"""
    t = np.linspace(0, 2 * np.pi, n)
    x = 20 + np.linspace(0, 400, n)
    y_top = 60 + amp * np.sin(t * 3)
    y_bot = 90 + amp * np.sin(t * 3 + 1.0)
    top = np.stack([x, y_top], axis=1)
    bot = np.stack([x[::-1], y_bot[::-1]], axis=1)
    return np.vstack([top, bot])


def test_snap_orthogonal_straightens_wavy_edges():
    pts = _wavy_ring()
    n = len(pts)
    snapped, straight, axis = snap_orthogonal(pts, closed=True, tol_deg=8.0, min_len=1.0)
    assert straight.any()
    # 每条直线链上的法向坐标应完全一致（拉成真直线）
    checked = 0
    for a in (1, 2):
        segs = np.where(axis == a)[0]
        runs, cur = [], []
        for i in segs:
            if cur and i == cur[-1] + 1:
                cur.append(int(i))
            else:
                if cur:
                    runs.append(cur)
                cur = [int(i)]
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) < 3:
                continue
            verts = [(i % n) for i in range(run[0], run[-1] + 2)]
            col = snapped[verts][:, 1] if a == 1 else snapped[verts][:, 0]
            assert float(col.max() - col.min()) < 1e-6
            checked += 1
    assert checked >= 2


def test_fit_contour_snap_emits_line_segments():
    pts = _wavy_ring()
    fc = fit_contour(pts, tolerance=0.5, max_nodes=60, snap_deg=8.0)
    line_segs = [s for s in fc.segs if s[1] is None]
    assert len(line_segs) >= 2        # 吸附后长边输出为直线段
    fc2 = fit_contour(pts, tolerance=0.5, max_nodes=60, snap_deg=0.0)
    assert fc.node_count <= fc2.node_count   # 直线段合并后节点不增


def test_waviness_metric_separates_wavy_and_straight():
    h, w = 160, 520
    straight = np.zeros((h, w), bool)
    straight[50:110, 20:460] = True
    wavy = np.zeros((h, w), bool)
    xs = np.arange(20, 460)
    for x in xs:
        y0 = 50 + int(round(2.0 * np.sin(x / 18.0)))
        wavy[y0:y0 + 60, x] = True
    src = straight.copy()
    rms_s, n_s = orthogonal_deviation(src, straight, contrast_img=np.where(src, 0, 255).astype(float))
    rms_w, n_w = orthogonal_deviation(src, wavy, contrast_img=np.where(src, 0, 255).astype(float))
    assert n_s >= 3 and n_w >= 3
    assert rms_w > rms_s + 0.2       # 波浪边的平直度显著更差


def test_curved_art_not_flagged_by_waviness():
    disc = np.zeros((200, 200), np.uint8)
    cv2.circle(disc, (100, 100), 60, 1, -1)
    m = disc.astype(bool)
    rms, n = orthogonal_deviation(m, m, contrast_img=np.where(m, 0, 255).astype(float))
    assert rms is None and n == 0     # 圆的边不是长直边，原图弯曲不误报


def _text_image(halo: str) -> np.ndarray:
    """白底黑字 + 柔和投影（blur）或硬边灰块（hard）。"""
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.putText(img, "TEXT", (28, 190), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (20, 20, 20), 12, cv2.LINE_AA)
    gray = np.full((300, 300), 255, np.uint8)
    cv2.putText(gray, "TEXT", (32, 194), cv2.FONT_HERSHEY_SIMPLEX, 3.2, 120, 12, cv2.LINE_AA)
    if halo == "blur":
        soft = cv2.GaussianBlur(gray, (0, 0), 6)
        region = soft < 250
        img[region] = np.minimum(img[region], soft[region][:, None])
    elif halo == "hard":
        region = (gray < 250)
        img[region] = 150
    return img


def test_blurred_halo_routed_to_shadow_layers():
    prepared = preprocess(_text_image("blur"))
    doc, stats = vectorize(prepared, VectorizeOptions.from_preset("balanced"))
    assert stats.get("shadow_layers", 0) >= 1      # 柔边投影独立成层（需求 5.4 保留）


def test_hard_gray_block_stays_in_main_layers():
    prepared = preprocess(_text_image("hard"))
    doc, stats = vectorize(prepared, VectorizeOptions.from_preset("balanced"))
    assert stats.get("shadow_layers", 0) == 0      # 硬边灰块是普通色层，不误判
