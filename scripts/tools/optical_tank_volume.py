"""Calculate optical floating-roof tank storage for ADD_with_metadata.

This command is independent of OlmoEarth/PyTorch because it operates on the
already split tank pairs. It writes one final CSV and no intermediate images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

try:
    from scripts.tools.optical_tank_geometry import (
        calculate_volume,
        detect_tank_circle,
        index_tank_pairs,
        measure_shadow,
        parse_optical_metadata,
        read_extend_px,
        read_rgb_u8,
        to_dict,
    )
except ModuleNotFoundError:
    from optical_tank_geometry import (
        calculate_volume,
        detect_tank_circle,
        index_tank_pairs,
        measure_shadow,
        parse_optical_metadata,
        read_extend_px,
        read_rgb_u8,
        to_dict,
    )


def discover_dates(data_root: Path, branches: list[str]) -> list[tuple[str, Path]]:
    dates: list[tuple[str, Path]] = []
    for branch in branches:
        branch_dir = data_root / branch
        if not branch_dir.is_dir():
            raise FileNotFoundError(branch_dir)
        for date_dir in sorted(path for path in branch_dir.iterdir() if path.is_dir()):
            if (date_dir / "no_shadow_split_tank").is_dir() and (
                date_dir / "shadow_split_tank"
            ).is_dir():
                dates.append((branch, date_dir))
    return dates


def candidate_rank(row: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(row.get("geometry_valid") is True),
        int(row.get("shadow_method") == "legacy_boundary_scan"),
        int(row.get("outer_edge_points", 0))
        + int(row.get("inner_edge_points", 0))
        + int(row.get("outer_arc_score", 0)),
    )


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    date_dirs = discover_dates(data_root, args.branches)
    if not date_dirs:
        raise FileNotFoundError(f"no optical date folders under {data_root}")

    indexes = [(branch, date, index_tank_pairs(date)) for branch, date in date_dirs]
    total = sum(len(pairs) for _, _, pairs in indexes)
    if args.limit is not None:
        total = min(total, args.limit)
    rows: list[dict[str, object]] = []

    with tqdm(total=total, desc="optical-volume") as progress:
        for branch, date_dir, pairs in indexes:
            info_dir = date_dir / "more_info"
            try:
                metadata = parse_optical_metadata(info_dir)
                extend_px = read_extend_px(info_dir)
            except (FileNotFoundError, ValueError) as exc:
                for tank_id, (no_path, shadow_candidates) in pairs.items():
                    if args.limit is not None and len(rows) >= args.limit:
                        break
                    rows.append(
                        {
                            "branch": branch,
                            "date": date_dir.name,
                            "tank_id": tank_id,
                            "no_shadow_file": no_path.name,
                            "shadow_file": shadow_candidates[0][0].name
                            if shadow_candidates
                            else "",
                            "geometry_valid": False,
                            "geometry_status": f"metadata_error: {exc}",
                        }
                    )
                    progress.update(1)
                continue

            for tank_id, (no_path, shadow_candidates) in pairs.items():
                if args.limit is not None and len(rows) >= args.limit:
                    break
                base: dict[str, object] = {
                    "branch": branch,
                    "date": date_dir.name,
                    "tank_id": tank_id,
                    "no_shadow_file": no_path.name,
                    "candidate_count": len(shadow_candidates),
                    "extend_px": extend_px,
                    **to_dict(metadata),
                }
                try:
                    circle = detect_tank_circle(read_rgb_u8(no_path))
                    circle_values = {
                        "circle_cx_px": circle.cx_px,
                        "circle_cy_no_shadow_px": circle.cy_px,
                        "circle_cy_shadow_px": circle.cy_px + extend_px,
                        "radius_px": circle.radius_px,
                        "radius_method": circle.method,
                        "circle_edge_support": circle.edge_support,
                    }
                except (OSError, ValueError, cv2.error) as exc:
                    rows.append(
                        {
                            **base,
                            "geometry_valid": False,
                            "geometry_status": f"image_error: {exc}",
                        }
                    )
                    progress.update(1)
                    continue

                if not shadow_candidates:
                    rows.append(
                        {
                            **base,
                            **circle_values,
                            "shadow_file": "",
                            "overlap_mode": "",
                            "geometry_valid": False,
                            "geometry_status": "missing_shadow_pair",
                        }
                    )
                    progress.update(1)
                    continue

                candidate_rows: list[dict[str, object]] = []
                for shadow_path, mode in shadow_candidates:
                    row = {
                        **base,
                        **circle_values,
                        "shadow_file": shadow_path.name,
                        "overlap_mode": mode,
                    }
                    try:
                        shadow = measure_shadow(
                            read_rgb_u8(shadow_path),
                            circle,
                            extend_px,
                            mode,
                            adjust_px=args.circle_adjust_px,
                        )
                        row.update(
                            {
                                "Lex_px": shadow.lex_px,
                                "Lin_px": shadow.lin_px,
                                "shadow_method": shadow.method,
                                "outer_edge_points": shadow.outer_edge_points,
                                "inner_edge_points": shadow.inner_edge_points,
                                "outer_arc_score": shadow.outer_arc_score,
                                "inner_arc_score": shadow.inner_arc_score,
                                **calculate_volume(circle, shadow, metadata),
                            }
                        )
                    except (OSError, ValueError, cv2.error) as exc:
                        row.update(
                            geometry_valid=False,
                            geometry_status=f"shadow_error: {exc}",
                        )
                    candidate_rows.append(row)
                rows.append(max(candidate_rows, key=candidate_rank))
                progress.update(1)
            if args.limit is not None and len(rows) >= args.limit:
                break

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    valid = sum(row.get("geometry_valid") is True for row in rows)
    print(f"wrote {out}  valid={valid}/{len(rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["ADD_with_metadata", "with_metadata"],
    )
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--circle-adjust-px", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
