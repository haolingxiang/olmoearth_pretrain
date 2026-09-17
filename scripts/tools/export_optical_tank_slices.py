"""Export unmarked tight tank chips, a clean attribute CSV, and a slice index map."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scripts.tools.extract_optical_tank_scene import extract_green_boxes, read_rgb

Image.MAX_IMAGE_PIXELS = None


def find_scene() -> tuple[Path, Path, Path]:
    for top in Path("E:/").iterdir():
        if not top.is_dir():
            continue
        hits = list(top.glob("186-1_*/L20/186-1.tif"))
        if not hits:
            continue
        image = hits[0]
        root = image.parents[1]
        det = next(root.glob("*.png"))
        csv_path = root / "oil_tank_features" / "oil_tank_features.csv"
        return image, det, csv_path
    raise FileNotFoundError("186-1 scene not found")


def write_slice_index(
    scene: Image.Image,
    crops: list[tuple[int, int, int, int, int]],
    chip_dir: Path,
    out_path: Path,
) -> None:
    """Overview of slice boxes on the original scene, plus numbered thumbnails."""
    arr = np.array(scene)
    xs1 = min(item[1] for item in crops)
    ys1 = min(item[2] for item in crops)
    xs2 = max(item[3] for item in crops)
    ys2 = max(item[4] for item in crops)
    margin = int(0.10 * max(xs2 - xs1, ys2 - ys1))
    xa, ya = max(0, xs1 - margin), max(0, ys1 - margin)
    xb, yb = min(arr.shape[1], xs2 + margin), min(arr.shape[0], ys2 + margin)
    farm = arr[ya:yb, xa:xb]
    target_w = 1800
    scale = target_w / farm.shape[1]
    overview = cv2.resize(
        farm,
        (target_w, max(1, int(round(farm.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    for tank_id, x1, y1, x2, y2 in crops:
        p1 = (int((x1 - xa) * scale), int((y1 - ya) * scale))
        p2 = (int((x2 - xa) * scale), int((y2 - ya) * scale))
        cv2.rectangle(overview, p1, p2, (0, 255, 80), 3)
        label = f"{tank_id:02d}"
        tx, ty = p1[0] + 4, max(28, p1[1] - 8)
        cv2.putText(overview, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overview, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = Image.fromarray(overview)

    thumb_h = 150
    gap = 8
    thumbs: list[Image.Image] = []
    for tank_id, *_rest in crops:
        chip = Image.open(chip_dir / f"tank_{tank_id:02d}.png").convert("RGB")
        ratio = thumb_h / chip.height
        thumb = chip.resize((max(1, int(chip.width * ratio)), thumb_h), Image.Resampling.LANCZOS)
        labeled = Image.new("RGB", (thumb.width, thumb.height + 22), (20, 20, 20))
        labeled.paste(thumb, (0, 22))
        tdraw = ImageDraw.Draw(labeled)
        tdraw.text((6, 4), f"{tank_id:02d}", fill=(255, 255, 255))
        thumbs.append(labeled)

    n_top = 7
    rows_img: list[Image.Image] = []
    for start in (0, n_top):
        row = thumbs[start : start + n_top]
        width = sum(im.width for im in row) + gap * (len(row) + 1)
        strip = Image.new("RGB", (width, thumb_h + 22 + 2 * gap), (20, 20, 20))
        x = gap
        for im in row:
            strip.paste(im, (x, gap))
            x += im.width + gap
        rows_img.append(strip)

    strip_w = max(im.width for im in rows_img)
    strip_h = sum(im.height for im in rows_img)
    thumbs_panel = Image.new("RGB", (strip_w, strip_h), (20, 20, 20))
    y = 0
    for im in rows_img:
        thumbs_panel.paste(im, ((strip_w - im.width) // 2, y))
        y += im.height

    final_w = max(canvas.width, thumbs_panel.width)
    final = Image.new("RGB", (final_w, canvas.height + thumbs_panel.height + 16), (12, 12, 12))
    final.paste(canvas, ((final_w - canvas.width) // 2, 0))
    final.paste(thumbs_panel, ((final_w - thumbs_panel.width) // 2, canvas.height + 16))
    final.save(out_path)


def main() -> None:
    image_path, det_path, csv_path = find_scene()
    frame = pd.read_csv(csv_path)
    overlay = read_rgb(det_path)
    scene = Image.open(image_path).convert("RGB")
    w_tif, h_tif = scene.size
    h_png, w_png = overlay.shape[:2]
    sx, sy = w_tif / w_png, h_tif / h_png
    boxes = extract_green_boxes(overlay)
    if len(boxes) != len(frame):
        raise ValueError(f"box/csv mismatch: {len(boxes)} vs {len(frame)}")

    out_dir = csv_path.parent.parent / "oil_tank_slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    crop_boxes: list[tuple[int, int, int, int, int]] = []
    for tank_id, (x, y, w, h) in enumerate(boxes):
        rec = frame.loc[frame["tank_id"] == tank_id].iloc[0]
        x1 = int(round(x * sx))
        y1 = int(round(y * sy))
        x2 = int(round((x + w) * sx))
        y2 = int(round((y + h) * sy))
        pad = int(round(0.06 * max(x2 - x1, y2 - y1)))
        xa, ya = max(0, x1 - pad), max(0, y1 - pad)
        xb, yb = min(w_tif, x2 + pad), min(h_tif, y2 + pad)
        crop_boxes.append((tank_id, xa, ya, xb, yb))
        chip = scene.crop((xa, ya, xb, yb))
        name = f"tank_{tank_id:02d}.png"
        chip.save(out_dir / name)
        rows.append(
            {
                "编号": tank_id,
                "切片": name,
                "经度": round(float(rec["lon"]), 6),
                "纬度": round(float(rec["lat"]), 6),
                "罐体直径_m": round(float(rec["tank_diameter_m"]), 2),
                "浮顶直径_m": round(float(rec["roof_diameter_m"]), 2),
                "罐高_m": round(float(rec["tank_height_m"]), 2),
                "浮顶下降高度_m": round(float(rec["roof_depth_m"]), 2),
                "液位_m": round(float(rec["oil_height_m"]), 2),
                "充装比例": round(float(rec["oil_storage_ratio"]), 3),
                "储量_m3": round(float(rec["V_m3"]), 1),
                "储量_bbl": round(float(rec["V_bbl"]), 1),
                "状态": rec["geometry_status"],
            }
        )
        print(name, chip.size)
    csv_out = out_dir / "oil_tank_attributes.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False, encoding="utf-8-sig")
    index_path = out_dir / "slice_index.png"
    write_slice_index(scene, crop_boxes, out_dir, index_path)
    print("wrote", csv_out)
    print("wrote", index_path)


if __name__ == "__main__":
    main()
