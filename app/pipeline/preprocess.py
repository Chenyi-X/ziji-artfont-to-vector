"""图像预处理与素材质量判定。

输入位图 -> 干净前景 + 质量报告。质量报告用于 5.6 节的无效输入边界判定。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityReport:
    width: int
    height: int
    subject_ratio: float          # 主体像素占全图比例
    background_uniform: bool      # 背景是否干净
    background_edge_noise: float  # 边缘区域颜色离散度
    blur_score: float             # 拉普拉斯方差，越低越糊
    glare_ratio: float            # 高光/过曝像素占比
    color_count: int              # 5bit 量化后的颜色数
    is_colorful: bool             # 是否多色/渐变
    subject_bbox_ratio: float = 0.0  # 主体外接矩形占画布比例（裁切后）
    orig_bbox_ratio: float = 0.0     # 裁切前主体外接矩形占画布比例（需求 5.6 口径）
    autocropped: bool = False
    strict_reject: bool = False       # 严格模式下应拒绝的硬伤
    severity: str = "ok"             # ok | warning | reject
    passed: bool = True
    invalid: bool = False             # 需求 5.6：默认模式也判出的素材异常（不阻断）
    invalid_reasons: list[str] = field(default_factory=list)
    photo_like: bool = False          # 疑似照片/写实图（非艺术字场景）
    textured_background: bool = False  # 纸张纹理/斑驳底纹（旧纸扫描件）
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "subject_ratio": round(self.subject_ratio, 4),
            "subject_bbox_ratio": round(self.subject_bbox_ratio, 4),
            "orig_bbox_ratio": round(self.orig_bbox_ratio, 4),
            "autocropped": self.autocropped,
            "strict_reject": self.strict_reject,
            "severity": self.severity,
            "background_uniform": self.background_uniform,
            "background_edge_noise": round(self.background_edge_noise, 3),
            "blur_score": round(self.blur_score, 2),
            "glare_ratio": round(self.glare_ratio, 4),
            "color_count": self.color_count,
            "is_colorful": self.is_colorful,
            "invalid": self.invalid,
            "invalid_reasons": self.invalid_reasons,
            "photo_like": self.photo_like,
            "textured_background": self.textured_background,
            "passed": self.passed,
            "warnings": self.warnings,
        }


@dataclass
class PreparedImage:
    rgba: np.ndarray              # 去背景后的 RGBA，背景 alpha=0
    mask: np.ndarray              # 前景硬掩膜 uint8 0/255
    soft_alpha: np.ndarray        # 前景软 alpha 0..255（保留抗锯齿边缘）
    quality: QualityReport
    skew_deg: float = 0.0
    rgb_raw: np.ndarray | None = None      # 未经 CLAHE/去噪的原始像素（颜色参考）
    bg_color: tuple[float, float, float] | None = None  # 估计的背景色 (RGB)
    input_shape: tuple[int, int] | None = None   # 输入原图 (h, w)，用于真值对齐
    canvas_shape: tuple[int, int] | None = None  # 校正/裁切前的处理画布 (h, w)
    rot_matrix: np.ndarray | None = None   # 倾斜校正矩阵（未校正时为 None）
    crop_box: tuple[int, int, int, int] | None = None  # 自动裁切 (x0, y0, x1, y1)
    source_rgba: np.ndarray | None = None  # 原始 RGB/alpha，仅作几何对齐

    def to_dict(self) -> dict:
        return {"quality": self.quality.to_dict(), "skew_deg": round(self.skew_deg, 2)}

    def align_mask(self, mask: np.ndarray) -> np.ndarray:
        """把与输入原图同尺寸的掩膜（如生成器真值字形）对齐到当前前景坐标系。

        按预处理顺序执行：输入原图尺寸 → 处理画布尺寸 → 倾斜校正旋转 → 自动裁切。
        """
        m = mask.astype(np.uint8)
        if self.input_shape is not None and m.shape[:2] != tuple(self.input_shape):
            m = cv2.resize(m, (self.input_shape[1], self.input_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        canvas = self.canvas_shape or self.mask.shape[:2]
        if m.shape[:2] != tuple(canvas):
            m = cv2.resize(m, (canvas[1], canvas[0]), interpolation=cv2.INTER_NEAREST)
        if self.rot_matrix is not None:
            m = cv2.warpAffine(m, self.rot_matrix, (canvas[1], canvas[0]),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if self.crop_box is not None:
            x0, y0, x1, y1 = self.crop_box
            m = m[y0:y1, x0:x1]
        return m


def _edge_pixels(rgb: np.ndarray, band: int = 3) -> np.ndarray:
    h, w = rgb.shape[:2]
    band = max(1, min(band, h // 4, w // 4))
    strips = [rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
              rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3)]
    return np.concatenate(strips, axis=0)


def _estimate_background(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """边界像素聚类估计背景色，返回背景色与边界颜色离散度。"""
    edge = _edge_pixels(rgb).astype(np.float32)
    # 用最亮/最常见的一簇作为背景：k=2 聚类取像素更多的簇
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    sample = edge[:: max(1, len(edge) // 4000)]
    cv2.setRNGSeed(20240614)          # 固定种子，保证同一张图多次处理结果一致
    _, labels, centers = cv2.kmeans(sample, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=2)
    bg = centers[int(np.argmax(counts))]
    spread = float(np.mean(np.linalg.norm(edge - bg, axis=1)))
    return bg, spread


def _dominant_colors(rgb: np.ndarray, core_mask: np.ndarray, k: int = 10,
                     min_share: float = 0.005) -> list[np.ndarray]:
    """取前景核心区的主色作为 alpha 估计的候选字色。

    占比门槛低到 0.5%：小面积装饰色（如卡通字旁的白色装饰点）虽然占比小，
    但是真正的前景色，不入选候选会让这类元素被当成背景丢掉。
    """
    pixels = rgb[core_mask > 0]
    if len(pixels) == 0:
        return []
    q = (pixels // 16).astype(np.int32)
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    out: list[np.ndarray] = []
    total = float(counts.sum())
    for i in order[: max(k * 3, k)]:
        if counts[i] / total < min_share:
            continue
        out.append(pixels[inverse == i].mean(axis=0).astype(np.float64))
        if len(out) >= k:
            break
    return out


def _flatten_paper_texture(rgb: np.ndarray, soft: np.ndarray,
                           mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """纸张底纹压平：剔除前景中占主导的大片近色纸面区块。

    旧纸扫描件（字标卡片、老广告）的纸面常被估成前景，霉斑与纸面同族色，
    会随纸面一起被描成上万节点的纹理层；真实业务里用户要的是纸上的字，
    纸面与斑溃应作背景扣除。用主色距离直方图的谷底定纸面容差，
    只剔除大面积连通区块，细笔画与小装饰保留。干净素材不触发。
    """
    fg = mask > 0
    if int(fg.sum()) < 800:
        return soft, mask, False
    ys, xs = np.nonzero(fg)
    bbox_fill = float(fg.sum()) / float(max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))
    if fg.mean() < 0.55 or bbox_fill < 0.85:
        # 前景没有铺满画面与外接矩形：字形/线稿主导，不是纸面
        return soft, mask, False
    px = rgb[fg].astype(np.float32)
    paper = np.median(px, axis=0)
    if float(paper.max()) < 90.0:
        # 纸面应是明亮色；主色本身是深墨/深色底时不压
        return soft, mask, False
    dist_core = np.linalg.norm(px - paper, axis=1)
    if float(np.percentile(dist_core, 50)) > 60.0:
        # 前景主色本身分散：正常多色/渐变素材，不压
        return soft, mask, False
    step = 6
    hist, edges = np.histogram(dist_core, bins=np.arange(0, 246, step))
    smooth = np.convolve(hist.astype(np.float64), np.ones(3) / 3.0, "same")
    lo, hi = int(30 // step) + 1, min(len(hist) - 2, int(150 // step))
    if hi <= lo:
        return soft, mask, False
    thr = min(float(edges[lo + int(np.argmin(smooth[lo:hi]))]), 90.0)
    mass_below = float((dist_core <= thr).mean())
    mass_above = float((dist_core >= thr + 15.0).mean())
    if thr < 40.0 or mass_below < 0.30 or mass_above < 0.015:
        # 纸面峰与字色之间没拉开距离、纸面不占主导、或纸面之外没有
        # 足够字色像素：不压
        return soft, mask, False
    dist_img = np.linalg.norm(rgb.astype(np.float32) - paper, axis=2)
    paper_mask = ((dist_img <= thr) & fg).astype(np.uint8)
    # 结构保护：笔画（含淡墨/褪色字）局部梯度明显，霉斑与纸面是平滑过渡；
    # 高梯度带及其邻域不参与压平，避免淡笔画被当底纹吃掉
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) / 4.0
    protect = cv2.dilate((mag > 14.0).astype(np.uint8),
                         np.ones((3, 3), np.uint8)) > 0
    paper_mask = (paper_mask & ~protect).astype(np.uint8)
    num, labels, st, _ = cv2.connectedComponentsWithStats(paper_mask, 8)
    if num <= 1:
        return soft, mask, False
    areas = st[1:, cv2.CC_STAT_AREA]
    big = np.zeros(num, dtype=bool)
    big[1:] = areas >= max(400.0, float(fg.sum()) * 0.003)
    flat = big[labels]
    if not flat.any():
        return soft, mask, False
    soft2 = np.where(flat, 0, soft).astype(np.uint8)
    mask2 = np.where(flat, 0, mask).astype(np.uint8)
    return soft2, mask2, True


def _estimate_alpha(rgb: np.ndarray, bg_color: np.ndarray,
                    edge_spread: float) -> np.ndarray:
    """亚像素前景覆盖估计：alpha = (P-B)·(C-B)/|C-B|²，多个候选字色取最大值。

    线性距离阈值会在笔画外多算约半个像素（描边偏胖、内孔容易发糊），
    这里用颜色混合模型直接解出覆盖率，50% 等值线就是真实字形边界。
    """
    bg = bg_color.astype(np.float64)
    dist = np.linalg.norm(rgb.astype(np.float64) - bg, axis=2)
    # 前景判据：与可见背景色差 >20（与样本真值生成同一口径）；
    # 噪声背景用边缘离散度自应抬高，避免把纹理当前景
    rough = dist > max(20.0, edge_spread * 2.6)
    core = cv2.erode(rough.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    if core.sum() < 50:
        core = rough
    candidates = _dominant_colors(rgb, core)
    if not candidates:
        thr = max(20.0, edge_spread * 2.6)
        return (np.clip((dist - thr) / max(12.0, thr * 0.6), 0, 1) * 255).astype(np.uint8)

    color = rgb.astype(np.float64)
    alpha = np.zeros(rgb.shape[:2], dtype=np.float64)
    for c in candidates:
        v = c - bg
        denom = float(v @ v)
        if denom < 100.0:
            continue
        a = ((color - bg) @ v) / denom
        np.maximum(alpha, a, out=alpha)
    return (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)


def _deskew_angle(mask: np.ndarray) -> float:
    """依靠最小外接矩形估计倾斜角，仅对小角度做校正。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    usable = [c.reshape(-1, 2) for c in contours if len(c) >= 5]
    if not usable:
        return 0.0
    pts = np.vstack(usable)
    if len(pts) < 20:
        return 0.0
    rect = cv2.minAreaRect(pts.astype(np.float32))
    angle = rect[-1]
    if angle > 45:
        angle -= 90
    if angle < -45:
        angle += 90
    return float(angle) if abs(angle) <= 8 else 0.0


def _quantized_color_count(rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, bool]:
    """统计 5bit/通道 量化后的颜色数，判断是否多色/渐变。"""
    q = (rgb // 8).astype(np.int32)
    if mask is not None:
        q = q[mask > 0]
    else:
        q = q.reshape(-1, 3)
    if q.size == 0:
        return 0, False
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    uniq = np.unique(keys)
    dominant = 0
    if uniq.size:
        counts = np.bincount(keys)
        top = np.sort(counts)[::-1][:3]
        dominant = int(top.sum())
    ratio = dominant / max(1, len(keys))
    return int(uniq.size), bool(uniq.size > 24 and ratio < 0.9)


def preprocess(image_bgr_or_rgba: np.ndarray,
               remove_background: bool = True,
               denoise: bool = True,
               enhance: bool = True,
               deskew: bool = True,
               max_side: int = 2400) -> PreparedImage:
    """主入口。

    传入 BGR（cv2 读取）或 BGRA/RGBA 均可；返回 PreparedImage。
    """
    data = image_bgr_or_rgba
    has_alpha = data.ndim == 3 and data.shape[2] == 4
    if has_alpha:
        bgr = data[:, :, :3].copy()
        alpha = data[:, :, 3].copy()
    else:
        bgr = data.copy()
        alpha = None

    # 带有真实透明区域的 PNG 才以 alpha 为准；alpha 全 255 的图走背景色判定
    transparent_ratio = float((alpha < 250).mean()) if alpha is not None else 0.0
    has_alpha = alpha is not None and transparent_ratio > 0.01

    h, w = bgr.shape[:2]
    orig_w, orig_h = w, h
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_raw = rgb.copy()   # 颜色指标的参考：CLAHE/去噪之前的原始像素
    source_rgba = np.dstack([rgb_raw, alpha if alpha is not None else np.full((h, w), 255, np.uint8)])
    raw_background, raw_spread = _estimate_background(rgb_raw)
    clean_source = has_alpha or raw_spread < 2.0
    if clean_source:
        enhance = False
        denoise = False

    if enhance:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l = clahe.apply(l)
        bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if denoise:
        gray_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr = cv2.bilateralFilter(bgr, 5, 40, 40)
        bgr = cv2.medianBlur(bgr, 3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        gray_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ---- 前景提取 ----
    bg_color, edge_spread = _estimate_background(rgb)
    rgb_raw_bg = tuple(float(v) for v in raw_background)
    stain_tol = 0.0
    textured = False
    if has_alpha:
        soft = alpha
        if denoise:
            soft = cv2.medianBlur(soft, 3)
    else:
        soft = _estimate_alpha(rgb, bg_color, edge_spread)

    if clean_source:
        mask = ((alpha > 127) if has_alpha else
                (np.linalg.norm(rgb_raw.astype(float)-raw_background, axis=2) > 20)).astype(np.uint8) * 255
    else:
        soft = cv2.GaussianBlur(soft, (3, 3), 0)
        mask = (soft > 127).astype(np.uint8) * 255

    # 保留最大连通区域集合，去掉背景噪点（小于总面积 0.01% 的碎片）
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels > 2 and not clean_source:
        areas = stats[1:, cv2.CC_STAT_AREA]
        total = mask.sum() / 255.0
        keep = np.zeros(n_labels, dtype=bool)
        keep[1:] = areas >= max(12.0, total * 0.0001)
        mask = np.where(keep[labels], 255, 0).astype(np.uint8)
        soft = np.where(mask > 0, soft, 0).astype(np.uint8)

        if mask.sum() == 0:  # 全部被判为噪点，回退
            mask = (soft > 127).astype(np.uint8) * 255

    # ---- 纸张纹理压平（旧纸扫描件） ----
    if remove_background and not has_alpha and not clean_source:
        soft, mask, textured = _flatten_paper_texture(rgb, soft, mask)

    if remove_background:
        rgba = np.dstack([bgr, soft]).copy()
    else:
        rgba = np.dstack([bgr, np.full((h, w), 255, np.uint8)]).copy()

    # ---- 倾斜校正 ----
    canvas_shape = (h, w)
    skew_deg = 0.0
    rot_matrix = None
    if deskew and not clean_source:
        skew_deg = _deskew_angle(mask)
        if abs(skew_deg) > 0.35:
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew_deg, 1.0)
            rgba = cv2.warpAffine(rgba, m, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))
            mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rgba[:, :, 3] = np.where(mask > 0, rgba[:, :, 3], 0)
            rgb_raw = cv2.warpAffine(rgb_raw, m, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            source_rgba = cv2.warpAffine(source_rgba, m, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            rot_matrix = m

    # ---- 自动裁切：主体在画布中占比过小时收紧到外接矩形 ----
    autocropped = False
    crop_box = None
    orig_bbox_ratio = 0.0
    ys, xs = np.nonzero(mask > 0)
    if len(xs):
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        orig_bbox_ratio = float(bw * bh) / float(h * w)
        if orig_bbox_ratio < 0.5:
            pad_x, pad_y = int(bw * 0.06) + 4, int(bh * 0.06) + 4
            x0, x1 = max(0, int(xs.min()) - pad_x), min(w, int(xs.max()) + pad_x + 1)
            y0, y1 = max(0, int(ys.min()) - pad_y), min(h, int(ys.max()) + pad_y + 1)
            rgba = rgba[y0:y1, x0:x1].copy()
            mask = mask[y0:y1, x0:x1].copy()
            soft = soft[y0:y1, x0:x1].copy()
            rgb_raw = rgb_raw[y0:y1, x0:x1].copy()
            source_rgba = source_rgba[y0:y1, x0:x1].copy()
            rgb = rgba[:, :, :3]
            h, w = mask.shape
            autocropped = True
            crop_box = (x0, y0, x1, y1)

    # ---- 质量报告 ----
    subject_ratio = float((mask > 0).sum()) / float(h * w)
    ys, xs = np.nonzero(mask > 0)
    if len(xs):
        bbox_ratio = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(h * w)
    else:
        bbox_ratio = 0.0
    # 模糊度只在主体外接矩形内计算，并在「前景合成到浅灰底」的图上算：
    # 透明底素材的 RGB 可能与字色完全相同（只靠 alpha 区分），直接取灰度会得到
    # 常数图；大片空白/透明区也会稀释全图拉普拉斯方差
    a_comp = soft.astype(np.float32)[:, :, None] / 255.0
    bgr_comp = rgba[:, :, :3].astype(np.float32)
    comp_gray_src = cv2.cvtColor(
        (bgr_comp * a_comp + 200.0 * (1 - a_comp)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ys_b, xs_b = np.nonzero(mask > 0)
    if len(xs_b):
        gy0, gy1 = max(0, int(ys_b.min()) - 10), min(comp_gray_src.shape[0], int(ys_b.max()) + 11)
        gx0, gx1 = max(0, int(xs_b.min()) - 10), min(comp_gray_src.shape[1], int(xs_b.max()) + 11)
        blur_region = comp_gray_src[gy0:gy1, gx0:gx1]
    else:
        blur_region = comp_gray_src
    blur_score = float(cv2.Laplacian(blur_region, cv2.CV_64F).var())
    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg_pixels = int((mask > 0).sum())
    glare_ratio = (float((gray_full[mask > 0] > 250).sum()) / fg_pixels) if fg_pixels else 0.0
    color_count, is_colorful = _quantized_color_count(rgb, cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2))
    # 透明底素材的“背景”是 alpha 通道，不存在纹理杂乱问题
    bg_uniform = True if has_alpha else edge_spread < 26.0

    report = QualityReport(
        width=w, height=h,
        subject_ratio=subject_ratio,
        background_uniform=bg_uniform,
        background_edge_noise=edge_spread,
        blur_score=blur_score,
        glare_ratio=glare_ratio,
        color_count=color_count,
        is_colorful=is_colorful,
        subject_bbox_ratio=bbox_ratio,
        autocropped=autocropped,
        orig_bbox_ratio=orig_bbox_ratio,
    )

    if fg_pixels == 0 or subject_ratio < 0.005:
        report.severity = "reject"
        report.passed = False
        report.warnings.append("基本没有识别到前景内容，请确认素材内容")
    strict_issues = []
    if autocropped:
        report.warnings.append("素材留白较多，已自动裁切到艺术字主体")
    invalid_reasons: list[str] = []
    if not bg_uniform:
        report.warnings.append("背景较为杂乱，已按边缘主色做背景扣除，效果可能受影响")
        strict_issues.append("background")
        invalid_reasons.append("background_complex")
    if blur_score < 25:
        report.warnings.append("图像较为模糊，笔画细节可能丢失")
        strict_issues.append("blur")
        invalid_reasons.append("blurry")
    if glare_ratio > 0.35:
        report.warnings.append("主体大面积过曝，颜色与细节可能失真")
        strict_issues.append("glare")
        invalid_reasons.append("glare")
    if min(orig_w, orig_h) < 500:
        report.warnings.append("分辨率低于 500×500，请更换清晰素材")
        strict_issues.append("resolution")
        invalid_reasons.append("resolution_small")
    # 资格判断使用裁切前占比，自动裁切不改变原始输入是否合规。
    if orig_bbox_ratio < 0.60:
        report.warnings.append(
            f"裁切前主体占画面 {orig_bbox_ratio * 100:.0f}%，低于 60% 输入规范，请更换素材")
        strict_issues.append("subject")
        invalid_reasons.append("subject_small")
    # 场景范围（需求二节）：照片/写实图不属于艺术字场景，给出提示
    photo_like = (bg_uniform is False and blur_score > 60
                  and color_count > 200 and subject_ratio > 0.35)
    if photo_like:
        report.warnings.append("疑似照片或写实图像，不属于艺术字素材范围，建议更换艺术字类素材")
        invalid_reasons.append("photo_like")

    # 纸张纹理底纹：不算无效素材（老字标/卡片扫描是真实业务场景），
    # 已按纹理压平处理，给提示并计入严格模式拒绝项
    if not has_alpha and textured:
        report.textured_background = True
        report.warnings.append("检测到斑驳纸张底纹，已按纹理压平处理")
        strict_issues.append("background")

    if fg_pixels == 0 or subject_ratio < 0.005:
        if "no_foreground" not in invalid_reasons:
            invalid_reasons.append("no_foreground")
    report.invalid_reasons = invalid_reasons
    report.invalid = bool(invalid_reasons)
    report.photo_like = photo_like

    report.strict_reject = bool(strict_issues)
    if report.severity != "reject":
        report.severity = "warning" if report.warnings else "ok"
    report.passed = report.severity != "reject"

    return PreparedImage(rgba=rgba, mask=mask, soft_alpha=soft,
                         quality=report, skew_deg=skew_deg,
                         rgb_raw=rgb_raw, bg_color=rgb_raw_bg,
                         input_shape=(orig_h, orig_w),
                         canvas_shape=canvas_shape,
                         rot_matrix=rot_matrix, crop_box=crop_box, source_rgba=source_rgba)
