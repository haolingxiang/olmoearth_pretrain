"""Optical floating-roof tank geometry ported from the legacy E:/code pipeline.

The implementation is deliberately in-memory: it does not write masks, Sobel
images, Hough visualisations or temporary spreadsheets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SHADOW_NAME_RE = re.compile(r"^(?P<tank>.+)_(?P<mode>[LNR])$", re.IGNORECASE)
LEGACY_SHADOW_TOLERANCE_PX = 5.0
DEFAULT_MIN_TANK_HEIGHT_M = 8.0
DEFAULT_MAX_TANK_HEIGHT_M = 25.0


@dataclass(frozen=True)
class OpticalMetadata:
    solar_elev_deg: float
    solar_az_deg: float
    sat_elev_deg: float
    sat_az_deg: float
    row_gsd_m: float
    column_gsd_m: float


@dataclass(frozen=True)
class CircleGeometry:
    cx_px: float
    cy_px: float
    radius_px: float
    method: str
    edge_support: float


@dataclass(frozen=True)
class ShadowGeometry:
    lex_px: float
    lin_px: float
    method: str
    valid: bool
    status: str
    outer_edge_points: int
    inner_edge_points: int
    outer_arc_score: int = 0
    inner_arc_score: int = 0
    quality_score: float = 0.0
    selection_reason: str = ""


def optical_height_factor(metadata: OpticalMetadata) -> float:
    solar_elev = math.radians(metadata.solar_elev_deg)
    sat_elev = math.radians(metadata.sat_elev_deg)
    azimuth_delta = math.radians(metadata.solar_az_deg - metadata.sat_az_deg)
    cot_solar = 1.0 / math.tan(solar_elev)
    cot_sat = 1.0 / math.tan(sat_elev)
    value = (
        cot_solar * cot_solar
        + cot_sat * cot_sat
        - 2.0 * cot_solar * cot_sat * math.cos(azimuth_delta)
    )
    return math.sqrt(max(value, 1e-8))


def calculate_volume(
    circle: CircleGeometry,
    shadow: ShadowGeometry,
    metadata: OpticalMetadata,
    min_tank_height_m: float = DEFAULT_MIN_TANK_HEIGHT_M,
    max_tank_height_m: float = DEFAULT_MAX_TANK_HEIGHT_M,
) -> dict[str, float | bool | str]:
    """Legacy physical model: the XML ImageRowGSD is the shared pixel scale."""
    scale = metadata.row_gsd_m
    radius_m = circle.radius_px * scale
    lex_m = shadow.lex_px * scale
    lin_m = shadow.lin_px * scale
    denominator = optical_height_factor(metadata)
    tank_height_m = lex_m / denominator
    roof_depth_m = lin_m / denominator
    raw_oil_height_m = tank_height_m - roof_depth_m
    # The legacy pipeline accepts Lin up to 5 px above Lex and treats that
    # case as an empty tank (zero oil-column height), not a missing result.
    oil_height_m = max(0.0, raw_oil_height_m)
    height_plausible = min_tank_height_m <= tank_height_m <= max_tank_height_m
    valid = (
        shadow.valid
        and radius_m > 0
        and tank_height_m > 0
        and roof_depth_m >= 0
        and height_plausible
    )
    max_volume_m3 = math.pi * radius_m**2 * tank_height_m if valid else float("nan")
    volume_m3 = math.pi * radius_m**2 * oil_height_m if valid else float("nan")
    if valid:
        status = "oil_empty" if raw_oil_height_m <= 0 else "ok"
    elif shadow.valid and not height_plausible:
        status = "implausible_tank_height"
    else:
        status = shadow.status
    return {
        "scale_used_m_per_px": scale,
        "R_m": radius_m,
        "Lex_m": lex_m,
        "Lin_m": lin_m,
        "tank_height_m": tank_height_m,
        "roof_depth_m": roof_depth_m,
        "raw_oil_height_m": raw_oil_height_m,
        "oil_height_m": oil_height_m,
        "oil_storage_ratio": oil_height_m / tank_height_m if valid else float("nan"),
        "max_volume_m3": max_volume_m3,
        "V_m3": volume_m3,
        "V_bbl": volume_m3 / 0.158987 if valid else float("nan"),
        "V_10k_bbl": volume_m3 / 1589.8 if valid else float("nan"),
        "geometry_valid": valid,
        "geometry_status": status,
        "height_plausible": height_plausible,
        "formula": "optical_shadow_eq1_2",
    }


def read_rgb_u8(path: Path) -> np.ndarray:
    """Read an image as uint8 RGB, including paths containing Chinese text."""
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_optical_metadata(info_dir: Path) -> OpticalMetadata:
    xml_paths = sorted(info_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"no XML metadata in {info_dir}")
    root = ET.parse(xml_paths[0]).getroot()

    def value(tag: str) -> float:
        text = root.findtext(f".//{tag}")
        if text is None:
            raise ValueError(f"missing {tag} in {xml_paths[0]}")
        return float(text)

    return OpticalMetadata(
        solar_elev_deg=value("SolarElevation"),
        solar_az_deg=value("SolarAzimuth"),
        sat_elev_deg=value("SatelliteElevation"),
        sat_az_deg=value("SatelliteAzimuth"),
        row_gsd_m=value("ImageRowGSD"),
        column_gsd_m=value("ImageColumnGSD"),
    )


def read_extend_px(info_dir: Path) -> int:
    path = info_dir / "extend_info.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(path.read_text(encoding="utf-8-sig").strip().splitlines()[0])


def split_shadow_name(path: Path) -> tuple[str, str] | None:
    match = SHADOW_NAME_RE.match(path.stem)
    if match is None:
        return None
    return match.group("tank"), match.group("mode").upper()


def index_tank_pairs(date_dir: Path) -> dict[str, tuple[Path, list[tuple[Path, str]]]]:
    """Pair ``cutX_Y.tif`` with one or more ``cutX_Y_[LNR].tif`` files."""
    no_dir = date_dir / "no_shadow_split_tank"
    shadow_dir = date_dir / "shadow_split_tank"
    no_shadow = {
        p.stem: p
        for p in sorted(no_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    candidates: dict[str, list[tuple[Path, str]]] = {}
    for path in sorted(shadow_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = split_shadow_name(path)
        if parsed is not None:
            tank, mode = parsed
            candidates.setdefault(tank, []).append((path, mode))
    return {
        tank: (path, candidates.get(tank, [])) for tank, path in no_shadow.items()
    }


def detect_tank_circle(rgb: np.ndarray) -> CircleGeometry:
    """Legacy CLAHE/Canny/Hough radius detector with a relaxed second pass."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(2, 2)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)
    edges = cv2.Canny(blurred, 160, 220)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.erode(cv2.dilate(edges, kernel, iterations=1), kernel, iterations=1)
    h, w = gray.shape

    def hough(param2: float, min_ratio: float) -> np.ndarray | None:
        return cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.6,
            minDist=300,
            param1=50,
            param2=param2,
            minRadius=max(4, int(w * min_ratio / 2)),
            maxRadius=max(5, int(w / 2)),
        )

    circles = hough(20, 0.8)
    method = "legacy_hough"
    if circles is None:
        circles = hough(16, 0.65)
        method = "relaxed_hough"
    if circles is None:
        return CircleGeometry(w / 2, h / 2, 0.0, "failed", 0.0)

    # The legacy implementation rounds every Hough candidate before both
    # scoring and returning it. Keeping OpenCV's sub-pixel radius here changes
    # the mask and propagates into both shadow measurements.
    rounded_circles = np.round(circles[0]).astype(int)
    best: tuple[int, int, int] | None = None
    best_support = -1.0
    for x, y, radius in rounded_circles:
        if not 0.3 * h < y < 0.7 * h:
            continue
        mask = np.zeros_like(edges)
        cv2.circle(mask, (int(x), int(y)), int(radius), 255, 1)
        support = float(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask)))
        if support > best_support:
            best = int(x), int(y), int(radius)
            best_support = support
    if best is None:
        return CircleGeometry(w / 2, h / 2, 0.0, "failed", 0.0)
    return CircleGeometry(float(best[0]), float(best[1]), float(best[2]), method, best_support)


def _split_inner_outer(
    rgb: np.ndarray, circle: CircleGeometry, extend_px: int, adjust_px: int
) -> tuple[np.ndarray, np.ndarray, int]:
    center_y = round(circle.cy_px + extend_px + adjust_px)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.circle(
        mask,
        (round(circle.cx_px), center_y),
        int(circle.radius_px * 1.05),
        255,
        -1,
    )
    white = np.full_like(rgb, 255)
    inside = np.where(mask[..., None] > 0, rgb, white)
    outside = np.where(mask[..., None] == 0, rgb, white)
    return inside, outside, center_y


def _dark_mask(rgb: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary)
    areas = [cv2.contourArea(c) for c in contours]
    mean_area = float(np.mean(areas))
    result = np.zeros_like(binary)
    total = rgb.shape[0] * rgb.shape[1]
    for contour, area in zip(contours, areas, strict=True):
        if area >= mean_area and area / total > 0.007:
            cv2.drawContours(result, [contour], -1, 255, cv2.FILLED)
    return result


def _outer_center(mask: np.ndarray, circle: CircleGeometry, extend_px: int) -> tuple[int, int]:
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chosen: tuple[int, int] | None = None
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        if mask.shape[1] / 5 < cx < mask.shape[1] * 4 / 5 and cy < mask.shape[0] * 2 / 3:
            if chosen is None or cy >= chosen[1]:
                chosen = cx, cy
    if chosen is not None:
        return chosen[0], chosen[1] - 3 if chosen[1] > 6 else chosen[1]
    return (
        round(circle.cx_px),
        round(circle.cy_px + extend_px - circle.radius_px - 15),
    )


def _inner_center(mask: np.ndarray) -> tuple[tuple[int, int], bool]:
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int] | None = None
    best_area = 0.0
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        area = cv2.contourArea(contour)
        central = mask.shape[1] / 5 < cx < mask.shape[1] * 4 / 5
        if central and cy > mask.shape[0] / 2 and area > best_area:
            bottom = max(point[0][1] for point in contour)
            best = cx, (cy + bottom) // 2 + 1
            best_area = area
    return ((0, 0), True) if best is None else (best, False)


def _sobel_edges(rgb: np.ndarray, threshold: int, diameter: int, iterations: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, diameter, 25 if diameter == 1 else 20, 75)
    gx = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 1, 0, ksize=3))
    gy = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 0, 1, ksize=3))
    edges = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    _, edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    return cv2.erode(dilated, kernel, iterations=iterations)


def _scan_inner(
    rgb: np.ndarray, center: tuple[int, int], threshold: int = 80
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    edges = _sobel_edges(rgb, threshold, diameter=1, iterations=1)
    h, w = edges.shape
    cx, cy = min(max(center[0], 2), w - 3), min(max(center[1], h // 2 + 1), h - 2)
    left = next(
        (
            j
            for j in range(cx, 2, -1)
            if int(edges[cy, j - 1]) - int(edges[cy, j]) > 200
        ),
        w // 5,
    )
    right = next(
        (
            j
            for j in range(cx, w - 2)
            if int(edges[cy, j]) - int(edges[cy, j - 1]) > 200
        ),
        w * 4 // 5,
    )
    upper: list[tuple[int, int]] = []
    lower: list[tuple[int, int]] = []
    for x in range(left, right):
        for y in range(cy, h // 2, -1):
            if int(edges[y - 1, x]) - int(edges[y, x]) > 200:
                upper.append((x, y - 1))
                break
        for y in range(cy, h - 1):
            if int(edges[y + 1, x]) - int(edges[y, x]) > 200 or y == h - 2:
                lower.append((x, y + 1))
                break
    return upper, lower


def _scan_outer(
    rgb: np.ndarray, center: tuple[int, int], threshold: int = 80
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    edges = _sobel_edges(rgb, threshold, diameter=2, iterations=2)
    h, w = edges.shape
    cx, cy = min(max(center[0], 3), w - 3), min(max(center[1], 6), h - 3)

    def find_left() -> int:
        for y in range(cy, 5, -1):
            for x in range(cx, 2, -1):
                if int(edges[y, x - 1]) - int(edges[y, x]) > 200:
                    return x
        return w // 5

    def find_right() -> int:
        for y in range(cy, 5, -1):
            for x in range(cx, w - 2):
                if int(edges[y, x]) - int(edges[y, x - 1]) > 200:
                    return x
        return w * 4 // 5

    upper: list[tuple[int, int]] = []
    lower: list[tuple[int, int]] = []
    for x in range(find_left(), find_right()):
        for y in range(cy, 2, -1):
            if int(edges[y - 1, x]) - int(edges[y, x]) > 200:
                upper.append((x, y - 2))
                break
        for y in range(cy, h - 2):
            if int(edges[y + 1, x]) - int(edges[y, x]) > 200:
                lower.append((x, y + 1))
                break
    return upper, lower


def _circumcircle(points: list[tuple[int, int]]) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = points
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-8:
        return None
    u1, u2, u3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (u1 * (y2 - y3) + u2 * (y3 - y1) + u3 * (y1 - y2)) / d
    cy = (u1 * (x3 - x2) + u2 * (x1 - x3) + u3 * (x2 - x1)) / d
    return cx, cy, math.hypot(x1 - cx, y1 - cy)


def _outer_named_points(
    upper: list[tuple[int, int]], lower: list[tuple[int, int]], mode: str, width: int
) -> tuple[tuple[float, float], tuple[int, int]] | None:
    if not upper or not lower:
        return None
    left, right = upper[0], upper[-1]
    if mode == "L":
        target_x = right[0] - 15
        middle = next(
            (point for point in upper if point[0] == target_x),
            upper[len(upper) // 2],
        )
    elif mode == "R":
        target_x = left[0] + 15
        middle = next(
            (point for point in upper if point[0] == target_x),
            upper[len(upper) // 2],
        )
    else:
        middle = min(upper, key=lambda p: abs(p[0] - width // 2))
        left = min(upper, key=lambda p: abs(p[0] - (width // 2 - 16)))
        right = min(upper, key=lambda p: abs(p[0] - (width // 2 + 16)))
    fitted = _circumcircle([left, middle, right])
    if fitted is None:
        return None
    cx, cy, radius = fitted
    named_upper = (cx, cy - radius)
    named_lower = min(lower, key=lambda p: abs(p[0] - cx))
    return named_upper, named_lower


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def shadow_height_m(shadow: ShadowGeometry, metadata: OpticalMetadata) -> float:
    """Convert the external shadow length to tank height in metres."""
    return shadow.lex_px * metadata.row_gsd_m / optical_height_factor(metadata)


def shadow_plausibility_reason(
    shadow: ShadowGeometry,
    metadata: OpticalMetadata,
    min_tank_height_m: float = DEFAULT_MIN_TANK_HEIGHT_M,
    max_tank_height_m: float = DEFAULT_MAX_TANK_HEIGHT_M,
) -> str | None:
    """Return why a shadow candidate is unusable, or ``None`` when plausible."""
    if not shadow.valid:
        return shadow.status
    if not math.isfinite(shadow.lex_px) or not math.isfinite(shadow.lin_px):
        return "non_finite_shadow"
    if shadow.lex_px <= 0 or shadow.lin_px < 0:
        return "invalid_shadow_length"
    if shadow.lex_px - shadow.lin_px < -LEGACY_SHADOW_TOLERANCE_PX:
        return "inner_shadow_exceeds_outer"
    height_m = shadow_height_m(shadow, metadata)
    if height_m < min_tank_height_m:
        return "tank_height_below_min"
    if height_m > max_tank_height_m:
        return "tank_height_above_max"
    return None


def _boundary_support_is_reliable(
    shadow: ShadowGeometry, circle: CircleGeometry
) -> bool:
    """Reject boundary fits supported by only a few noisy edge pixels."""
    minimum = max(6, math.ceil(circle.radius_px * 0.2))
    if shadow.outer_edge_points < minimum:
        return False
    return shadow.lin_px == 0 or shadow.inner_edge_points >= minimum


def _shadow_quality(shadow: ShadowGeometry, circle: CircleGeometry) -> float:
    scale = max(1.0, circle.radius_px * 2.0)
    if shadow.method == "legacy_boundary_scan":
        outer = min(1.0, shadow.outer_edge_points / scale)
        inner = 1.0 if shadow.lin_px == 0 else min(1.0, shadow.inner_edge_points / scale)
        return 0.6 * outer + 0.4 * inner
    outer = min(1.0, shadow.outer_arc_score / scale)
    inner = 1.0 if shadow.lin_px == 0 else min(1.0, shadow.inner_arc_score / scale)
    return 0.65 * outer + 0.35 * inner


def select_shadow_candidate(
    boundary: ShadowGeometry,
    arc: ShadowGeometry,
    circle: CircleGeometry,
    metadata: OpticalMetadata,
    min_tank_height_m: float = DEFAULT_MIN_TANK_HEIGHT_M,
    max_tank_height_m: float = DEFAULT_MAX_TANK_HEIGHT_M,
) -> ShadowGeometry:
    """Choose a physically plausible shadow result with sufficient evidence.

    Boundary scanning remains preferred because it is more accurate on this
    dataset, but it is no longer accepted merely because it returned numbers.
    Implausible height and weak edge support trigger the arc fallback.
    """
    boundary_reason = shadow_plausibility_reason(
        boundary, metadata, min_tank_height_m, max_tank_height_m
    )
    boundary_supported = _boundary_support_is_reliable(boundary, circle)
    boundary_quality = _shadow_quality(boundary, circle)
    if boundary_reason is None and boundary_supported:
        return replace(
            boundary,
            quality_score=boundary_quality,
            selection_reason="boundary_plausible_and_supported",
        )

    arc_reason = shadow_plausibility_reason(
        arc, metadata, min_tank_height_m, max_tank_height_m
    )
    arc_quality = _shadow_quality(arc, circle)
    if arc_reason is None:
        reason = boundary_reason or "boundary_edge_support_too_low"
        return replace(
            arc,
            quality_score=arc_quality,
            selection_reason=f"arc_fallback:{reason}",
        )

    if boundary_reason is None:
        rejected = replace(
            boundary,
            valid=False,
            status="shadow_low_confidence",
            quality_score=boundary_quality,
            selection_reason=f"boundary_edge_support_too_low;arc:{arc_reason}",
        )
        return rejected

    selected = boundary if boundary_quality >= arc_quality else arc
    return replace(
        selected,
        valid=False,
        status="shadow_not_plausible",
        quality_score=max(boundary_quality, arc_quality),
        selection_reason=f"boundary:{boundary_reason};arc:{arc_reason}",
    )


def _arc_match(
    rgb: np.ndarray, circle: CircleGeometry, extend_px: int
) -> tuple[float, float, int, int]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, 2, 20, 75)
    gx = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 1, 0, ksize=3))
    gy = cv2.convertScaleAbs(cv2.Sobel(filtered, cv2.CV_16S, 0, 1, ksize=3))
    edge = cv2.addWeighted(gx, 0.1, gy, 0.9, 0)
    _, edge = cv2.threshold(edge, 50, 255, cv2.THRESH_BINARY)
    cx = round(circle.cx_px)
    # ``adjust_px`` moves only the crop mask. Applying it again to the arc
    # origin systematically shifted both matched boundaries in the v2 run.
    base_y = round(circle.cy_px + extend_px)
    radius = round(circle.radius_px)

    def best_offset(
        offsets: range, arcs: tuple[tuple[int, int], tuple[int, int]]
    ) -> tuple[int, int]:
        best_score, best = 0, 0
        for offset in offsets:
            arc = np.zeros_like(edge)
            for start, end in arcs:
                cv2.ellipse(arc, (cx, base_y + offset), (radius, radius), 0, start, end, 255, 2)
            score = int(np.count_nonzero(cv2.bitwise_and(arc, edge)))
            if score > best_score:
                best_score, best = score, offset
        return best, best_score

    out_offset, out_score = best_offset(range(-50, -10), ((220, 270), (270, 320)))
    start, stop = int(-radius * 0.8), int(-radius * 0.1)
    in_offset, in_score = best_offset(range(start, max(start + 1, stop)), ((60, 90), (90, 120)))
    if in_score < 50:
        in_offset = 0
    return float(-out_offset), float(-in_offset), out_score, in_score


def measure_shadow(
    rgb: np.ndarray,
    circle: CircleGeometry,
    metadata: OpticalMetadata,
    extend_px: int,
    overlap_mode: str,
    adjust_px: int = 5,
    min_tank_height_m: float = DEFAULT_MIN_TANK_HEIGHT_M,
    max_tank_height_m: float = DEFAULT_MAX_TANK_HEIGHT_M,
) -> ShadowGeometry:
    """Measure both shadow boundaries and reject implausible geometry."""
    if circle.radius_px <= 0:
        return ShadowGeometry(0, 0, "none", False, "circle_not_found", 0, 0)
    inside, outside, _ = _split_inner_outer(rgb, circle, extend_px, adjust_px)
    out_center = _outer_center(_dark_mask(outside, 60), circle, extend_px)
    out_upper, out_lower = _scan_outer(outside, out_center, 80)
    out_named = _outer_named_points(out_upper, out_lower, overlap_mode, rgb.shape[1])
    inner_center, full = _inner_center(_dark_mask(inside, 100))

    inner_upper: list[tuple[int, int]] = []
    inner_lower: list[tuple[int, int]] = []
    lex = lin = 0.0
    boundary_valid = False
    if out_named is not None:
        out_up, out_down = out_named
        lex = _distance(out_up, out_down)
        if full:
            lin = 0.0
        else:
            inner_upper, inner_lower = _scan_inner(inside, inner_center, 80)
            if inner_upper and inner_lower:
                in_up = min(inner_upper, key=lambda p: abs(p[0] - out_up[0]))
                in_down = min(inner_lower, key=lambda p: abs(p[0] - out_up[0]))
                lin = _distance(in_up, in_down)
        boundary_valid = (
            out_up[0] != 0
            and out_up[1] > -5
            and lex > 0
            and lin >= 0
            and lex - lin >= -LEGACY_SHADOW_TOLERANCE_PX
        )
    boundary = ShadowGeometry(
        lex,
        lin,
        "legacy_boundary_scan",
        boundary_valid,
        "ok" if boundary_valid else "boundary_not_found",
        len(out_upper),
        len(inner_upper),
    )

    arc_lex, arc_lin, out_score, in_score = _arc_match(rgb, circle, extend_px)
    arc_valid = (
        out_score > 0
        and arc_lex > 0
        and arc_lin >= 0
        and arc_lex - arc_lin >= -LEGACY_SHADOW_TOLERANCE_PX
    )
    arc = ShadowGeometry(
        arc_lex,
        arc_lin,
        "legacy_arc_match",
        arc_valid,
        "ok" if arc_valid else "arc_not_found",
        len(out_upper),
        len(inner_upper),
        out_score,
        in_score,
    )
    return select_shadow_candidate(
        boundary,
        arc,
        circle,
        metadata,
        min_tank_height_m,
        max_tank_height_m,
    )


def to_dict(value: object) -> dict[str, object]:
    return asdict(value)  # type: ignore[arg-type]
