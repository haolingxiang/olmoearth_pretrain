"""Extract optical floating-roof tank attributes from a detected scene.

Uses detection boxes (green overlay) plus the georeferenced RGB GeoTIFF.
Planar attributes come from circle/roof fits. Height and storage follow the
existing optical shadow model in optical_tank_geometry.py.

The scene GeoTIFF has no solar/satellite XML. Default: nadir (sat elev 90 deg)
and a leaf-on mid-latitude solar elevation. Solar azimuth is estimated from
the outer-shadow crescent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from scripts.tools.optical_tank_geometry import (
    CircleGeometry,
    OpticalMetadata,
    ShadowGeometry,
    calculate_volume,
)

Image.MAX_IMAGE_PIXELS = None

GSD_M = 0.297728
UL_LON = -96.7558429300
UL_LAT = 35.9632392900
X_DEG = 0.0000026822
Y_DEG = -0.0000026825


def read_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        rgb = np.array(Image.open(path).convert("RGB"))
        if rgb.ndim != 3:
            raise ValueError(f"cannot read RGB image: {path}")
        return rgb
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_green_boxes(overlay: np.ndarray) -> list[tuple[int, int, int, int]]:
    r, g, b = overlay[:, :, 0], overlay[:, :, 1], overlay[:, :, 2]
    mask = ((g > 150) & (g > r + 60) & (g > b + 60)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 50:
            continue
        if not 0.7 <= w / max(h, 1) <= 1.4:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def pixel_to_lonlat(col: float, row: float) -> tuple[float, float]:
    return UL_LON + col * X_DEG, UL_LAT + row * Y_DEG


def fit_roof_near_center(rgb: np.ndarray, cx: float, cy: float, search_r: float) -> CircleGeometry:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    roi = dist <= search_r
    if not np.any(roi):
        return CircleGeometry(cx, cy, 0.0, "failed", 0.0)
    # Pontoon roofs are mottled; a high percentile keeps only bright cells.
    thresh = float(np.percentile(gray[roi], 48))
    roof = (roi & (gray >= thresh)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    roof = cv2.morphologyEx(roof, cv2.MORPH_CLOSE, kernel)
    roof = cv2.morphologyEx(roof, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(roof, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return CircleGeometry(cx, cy, 0.0, "failed", 0.0)
    best = None
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < math.pi * (search_r * 0.18) ** 2:
            continue
        (rx, ry), radius = cv2.minEnclosingCircle(contour)
        if radius > search_r * 1.05:
            continue
        center_pen = math.hypot(rx - cx, ry - cy)
        score = area - 8.0 * center_pen
        if score > best_score:
            best = CircleGeometry(float(rx), float(ry), float(radius), "bright_disk", area)
            best_score = score
    return best if best is not None else CircleGeometry(cx, cy, 0.0, "failed", 0.0)


def choose_tank_circle(rgb: np.ndarray, box_w_px: float, box_h_px: float) -> CircleGeometry:
    """Tank centre follows the detection box; radius follows the roof plus wall."""
    h, w = rgb.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0
    box_r = 0.5 * min(box_w_px, box_h_px)
    roof = fit_roof_near_center(rgb, cx0, cy0, search_r=0.58 * box_r)
    if roof.radius_px <= 0:
        return CircleGeometry(cx0, cy0, 0.52 * box_r, "box_prior", 0.0)
    wall_r = float(np.clip(roof.radius_px * 1.12, 0.40 * box_r, 0.66 * box_r))
    return CircleGeometry(cx0, cy0, wall_r, "roof_expanded", roof.edge_support)


def fit_roof_circle(rgb: np.ndarray, tank: CircleGeometry) -> CircleGeometry:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - tank.cx_px) ** 2 + (yy - tank.cy_px) ** 2)
    inner = dist <= tank.radius_px * 0.98
    if not np.any(inner):
        return CircleGeometry(tank.cx_px, tank.cy_px, tank.radius_px * 0.9, "tank_scaled", 0.0)
    thresh = float(np.percentile(gray[inner], 46))
    roof = (inner & (gray >= thresh)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    roof = cv2.morphologyEx(roof, cv2.MORPH_CLOSE, kernel)
    roof = cv2.morphologyEx(roof, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(roof, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return CircleGeometry(tank.cx_px, tank.cy_px, tank.radius_px * 0.9, "tank_scaled", 0.0)
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    radius = float(min(radius, tank.radius_px * 0.98))
    if radius < tank.radius_px * 0.55:
        return CircleGeometry(tank.cx_px, tank.cy_px, tank.radius_px * 0.9, "tank_scaled", 0.0)
    return CircleGeometry(float(cx), float(cy), radius, "bright_disk", float(cv2.contourArea(contour)))


def polar_bin_means(
    gray: np.ndarray, cx: float, cy: float, r0: float, r1: float, n_bins: int = 36
) -> np.ndarray:
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    dx = xx - cx
    dy = cy - yy  # image y-down -> north-up
    dist = np.sqrt(dx * dx + dy * dy)
    azimuth = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    ring = (dist >= r0) & (dist <= r1)
    means = np.full(n_bins, np.nan, dtype=np.float64)
    bin_w = 360.0 / n_bins
    for i in range(n_bins):
        sel = ring & (azimuth >= i * bin_w) & (azimuth < (i + 1) * bin_w)
        if np.any(sel):
            means[i] = float(gray[sel].mean())
    return means


def scene_shadow_azimuth_deg(gray_crops: list[tuple[np.ndarray, CircleGeometry]]) -> float:
    acc = np.zeros(36, dtype=np.float64)
    count = np.zeros(36, dtype=np.float64)
    for gray, circle in gray_crops:
        means = polar_bin_means(
            gray, circle.cx_px, circle.cy_px, circle.radius_px * 1.02, circle.radius_px * 1.38
        )
        valid = np.isfinite(means)
        acc[valid] += means[valid]
        count[valid] += 1
    avg = np.divide(acc, np.maximum(count, 1), out=np.full_like(acc, np.nan), where=count > 0)
    if not np.any(np.isfinite(avg)):
        return 0.0
    return float(np.nanargmin(avg) * 10.0 + 5.0)


def sample_radial(
    gray: np.ndarray, cx: float, cy: float, az_deg: float, r0: float, r1: float
) -> tuple[np.ndarray, np.ndarray]:
    # az_deg: geographic, clockwise from north; image x east, y south
    rad = math.radians(az_deg)
    dx, dy = math.sin(rad), -math.cos(rad)
    radii = np.arange(r0, r1, 1.0)
    xs = cx + dx * radii
    ys = cy + dy * radii
    h, w = gray.shape
    inside = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
    vals = np.full(radii.shape, np.nan, dtype=np.float64)
    for i, ok in enumerate(inside):
        if not ok:
            continue
        vals[i] = float(
            cv2.getRectSubPix(gray, (1, 1), (float(xs[i]), float(ys[i])))[0, 0]
        )
    return radii, vals


def dark_run_length(values: np.ndarray, dark: np.ndarray) -> float:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return 0.0
    finite = np.isfinite(values)
    dark = dark & finite
    length = 0
    started = False
    for flag in dark:
        if flag:
            started = True
            length += 1
        elif started:
            break
    return float(length)


def _finite_median(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.median(finite))


def measure_one_azimuth(
    gray: np.ndarray, tank: CircleGeometry, roof: CircleGeometry, az_deg: float
) -> tuple[float, float, int, int]:
    r = tank.radius_px
    _, inner_vals = sample_radial(gray, tank.cx_px, tank.cy_px, az_deg, max(1.0, r * 0.25), r * 0.99)
    _, outer_vals = sample_radial(gray, tank.cx_px, tank.cy_px, az_deg, r * 0.98, r * 1.55)
    roof_level = _finite_median(inner_vals[: max(3, inner_vals.size // 3)], 180.0)
    berm_level = _finite_median(outer_vals[-max(3, outer_vals.size // 4) :], 140.0)
    thresh = min(80.0, 0.35 * roof_level + 0.15 * berm_level)
    outer_dark = np.isfinite(outer_vals) & (outer_vals <= thresh)
    inner_dark = np.isfinite(inner_vals) & (inner_vals <= thresh)
    lex = dark_run_length(outer_vals, outer_dark)
    lin = dark_run_length(inner_vals[::-1], inner_dark[::-1])
    if roof.radius_px > 0:
        ring = max(0.0, tank.radius_px - roof.radius_px)
        if lin < 2 and ring > 2:
            lin = ring
        lin = min(lin, tank.radius_px * 0.85)
    return lex, lin, int(np.count_nonzero(outer_dark)), int(np.count_nonzero(inner_dark))


def measure_shadow_radial(
    rgb: np.ndarray, tank: CircleGeometry, roof: CircleGeometry, shadow_az_deg: float
) -> ShadowGeometry:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    best = None
    for delta in range(-15, 16, 5):
        az = (shadow_az_deg + delta) % 360.0
        lex, lin, n_out, n_in = measure_one_azimuth(gray, tank, roof, az)
        cand = (lex, -abs(delta), lin, n_out, n_in, az)
        if best is None or cand[:2] > best[:2]:
            best = cand
    assert best is not None
    lex, _, lin, n_out, n_in, az = best
    valid = lex >= 2
    if valid and lin > lex + 2:
        lin = lex
    return ShadowGeometry(
        lex_px=lex,
        lin_px=lin,
        method="radial_profile",
        valid=valid,
        status="ok" if valid else "outer_shadow_not_found",
        outer_edge_points=n_out,
        inner_edge_points=n_in,
        quality_score=float(lex) + 0.3 * float(lin),
        selection_reason=f"shadow_az={az:.1f}",
    )


def circle_polygon(cx: float, cy: float, radius: float, n: int = 64) -> list[list[float]]:
    pts = []
    for i in range(n):
        ang = 2.0 * math.pi * i / n
        lon, lat = pixel_to_lonlat(cx + radius * math.cos(ang), cy + radius * math.sin(ang))
        pts.append([lon, lat])
    pts.append(pts[0])
    return pts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--det-png", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gsd", type=float, default=GSD_M)
    parser.add_argument("--solar-elev", type=float, default=50.0)
    parser.add_argument("--sat-elev", type=float, default=90.0)
    parser.add_argument("--sat-az", type=float, default=180.0)
    parser.add_argument("--crop-pad", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlay = read_rgb(args.det_png)
    scene = np.array(Image.open(args.image).convert("RGB"))
    h_png, w_png = overlay.shape[:2]
    h_tif, w_tif = scene.shape[:2]
    sx, sy = w_tif / w_png, h_tif / h_png
    boxes = extract_green_boxes(overlay)
    if not boxes:
        raise ValueError(f"no tank boxes found in {args.det_png}")

    crops: list[tuple[int, np.ndarray, CircleGeometry, tuple[int, int, int, int]]] = []
    for i, (x, y, w, h) in enumerate(boxes):
        x1 = int(round(x * sx))
        y1 = int(round(y * sy))
        x2 = int(round((x + w) * sx))
        y2 = int(round((y + h) * sy))
        box_w, box_h = float(x2 - x1), float(y2 - y1)
        pad = int(round(args.crop_pad * max(box_w, box_h)))
        xa, ya = max(0, x1 - pad), max(0, y1 - pad)
        xb, yb = min(w_tif, x2 + pad), min(h_tif, y2 + pad)
        crop = scene[ya:yb, xa:xb]
        circle = choose_tank_circle(crop, box_w, box_h)
        crops.append((i, crop, circle, (xa, ya, xb, yb)))

    gray_for_az = [
        (cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), circle) for _, crop, circle, _ in crops
    ]
    shadow_az = scene_shadow_azimuth_deg(gray_for_az)
    solar_az = (shadow_az + 180.0) % 360.0
    metadata = OpticalMetadata(
        solar_elev_deg=args.solar_elev,
        solar_az_deg=solar_az,
        sat_elev_deg=args.sat_elev,
        sat_az_deg=args.sat_az,
        row_gsd_m=args.gsd,
        column_gsd_m=args.gsd,
    )

    out_dir = args.out_dir
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    overlay_scene = scene.copy()

    for tank_id, crop, tank, (xa, ya, xb, yb) in crops:
        roof = fit_roof_circle(crop, tank)
        shadow = measure_shadow_radial(crop, tank, roof, shadow_az)
        volume = calculate_volume(tank, shadow, metadata)
        cx = xa + tank.cx_px
        cy = ya + tank.cy_px
        lon, lat = pixel_to_lonlat(cx, cy)
        roof_cx = xa + roof.cx_px
        roof_cy = ya + roof.cy_px
        row = {
            "tank_id": tank_id,
            "lon": lon,
            "lat": lat,
            "pixel_x": cx,
            "pixel_y": cy,
            "bbox_x1": xa,
            "bbox_y1": ya,
            "bbox_x2": xb,
            "bbox_y2": yb,
            "tank_radius_px": tank.radius_px,
            "tank_diameter_m": 2.0 * tank.radius_px * args.gsd,
            "tank_outline_method": tank.method,
            "roof_radius_px": roof.radius_px,
            "roof_diameter_m": 2.0 * roof.radius_px * args.gsd,
            "roof_cx": roof_cx,
            "roof_cy": roof_cy,
            "shadow_az_deg": shadow_az,
            "solar_az_deg": solar_az,
            "solar_elev_deg_assumed": args.solar_elev,
            "sat_elev_deg_assumed": args.sat_elev,
            "Lex_px": shadow.lex_px,
            "Lin_px": shadow.lin_px,
            "shadow_method": shadow.method,
            **volume,
        }
        rows.append(row)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                    for k, v in row.items()
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [circle_polygon(cx, cy, tank.radius_px)],
                },
            }
        )

        vis = crop.copy()
        cv2.circle(vis, (int(tank.cx_px), int(tank.cy_px)), int(tank.radius_px), (0, 255, 255), 2)
        if roof.radius_px > 0:
            cv2.circle(vis, (int(roof.cx_px), int(roof.cy_px)), int(roof.radius_px), (255, 220, 0), 2)
        ray = math.radians(shadow_az)
        x2 = int(tank.cx_px + math.sin(ray) * tank.radius_px * 1.5)
        y2 = int(tank.cy_px - math.cos(ray) * tank.radius_px * 1.5)
        cv2.arrowedLine(vis, (int(tank.cx_px), int(tank.cy_px)), (x2, y2), (255, 0, 255), 2, tipLength=0.08)
        label = f"#{tank_id} D={row['tank_diameter_m']:.1f}m"
        cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        Image.fromarray(vis).save(crop_dir / f"tank_{tank_id:02d}.png")

        cv2.circle(overlay_scene, (int(cx), int(cy)), int(tank.radius_px), (0, 255, 255), 3)
        if roof.radius_px > 0:
            cv2.circle(overlay_scene, (int(roof_cx), int(roof_cy)), int(roof.radius_px), (255, 220, 0), 2)
        cv2.putText(
            overlay_scene,
            str(tank_id),
            (int(cx) - 10, int(cy) - int(tank.radius_px) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "oil_tank_features.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        {
            "编号": frame["tank_id"],
            "经度": frame["lon"],
            "纬度": frame["lat"],
            "罐体直径_m": frame["tank_diameter_m"],
            "浮顶直径_m": frame["roof_diameter_m"],
            "罐高_m": frame["tank_height_m"],
            "浮顶下降高度_m": frame["roof_depth_m"],
            "液位_m": frame["oil_height_m"],
            "充装比例": frame["oil_storage_ratio"],
            "储量_m3": frame["V_m3"],
            "储量_bbl": frame["V_bbl"],
            "几何有效": frame["geometry_valid"],
            "状态": frame["geometry_status"],
        }
    )
    summary_path = out_dir / "oil_tank_features_zh.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "metadata": {
            "image": str(args.image),
            "det_png": str(args.det_png),
            "gsd_m": args.gsd,
            "solar_elev_deg_assumed": args.solar_elev,
            "sat_elev_deg_assumed": args.sat_elev,
            "shadow_az_deg": shadow_az,
            "formula": "optical_shadow_eq1_2",
        },
        "features": features,
    }
    (out_dir / "oil_tank_features.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    preview = cv2.resize(
        overlay_scene,
        (overlay.shape[1], overlay.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    Image.fromarray(preview).save(out_dir / "oil_tank_features_overlay.png")
    print(f"tanks={len(rows)} shadow_az={shadow_az:.1f} solar_az={solar_az:.1f}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
