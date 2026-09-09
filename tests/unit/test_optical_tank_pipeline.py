"""Unit tests for the optical tank dataset adapter and physical model."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from scripts.tools.optical_tank_geometry import (
    _near_best_height_index,
    CircleGeometry,
    OpticalMetadata,
    ShadowGeometry,
    calculate_volume,
    index_tank_pairs,
    parse_optical_metadata,
    read_extend_px,
    select_shadow_candidate,
    shadow_plausibility_reason,
    split_shadow_name,
    tank_height_shadow_bounds_px,
)


class OpticalTankPipelineTests(unittest.TestCase):
    def test_near_best_arc_score_uses_height_only_as_tiebreaker(self) -> None:
        # 95 is within 80% of the best score, so its more central physical
        # height wins. A score of 79 is outside the near-best set even though
        # its height is exactly at the midpoint.
        selected = _near_best_height_index(
            [100.0, 95.0, 79.0],
            [23.0, 17.0, 16.5],
            midpoint_height_m=16.5,
        )
        self.assertEqual(selected, 1)

    def test_lnr_name_is_paired_to_base_tank(self) -> None:
        self.assertEqual(split_shadow_name(Path("cut10_1_L.tif")), ("cut10_1", "L"))
        self.assertEqual(split_shadow_name(Path("cut10_2_n.tiff")), ("cut10_2", "N"))
        self.assertIsNone(split_shadow_name(Path("cut10_3.tif")))

    def test_date_pair_index_accepts_multiple_shadow_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            date = Path(tmp)
            no_dir = date / "no_shadow_split_tank"
            shadow_dir = date / "shadow_split_tank"
            no_dir.mkdir()
            shadow_dir.mkdir()
            (no_dir / "cut1_1.tif").touch()
            (shadow_dir / "cut1_1_L.tif").touch()
            (shadow_dir / "cut1_1_N.tif").touch()
            indexed = index_tank_pairs(date)
        self.assertEqual(len(indexed["cut1_1"][1]), 2)
        self.assertEqual({mode for _, mode in indexed["cut1_1"][1]}, {"L", "N"})

    def test_xml_metadata_and_extend_info(self) -> None:
        xml = """<root><SolarElevation>45</SolarElevation><SolarAzimuth>100</SolarAzimuth>
        <SatelliteElevation>70</SatelliteElevation><SatelliteAzimuth>280</SatelliteAzimuth>
        <ImageRowGSD>1.08</ImageRowGSD><ImageColumnGSD>1.17</ImageColumnGSD></root>"""
        with tempfile.TemporaryDirectory() as tmp:
            info = Path(tmp)
            (info / "meta.xml").write_text(xml, encoding="utf-8")
            (info / "extend_info.txt").write_text("40\n", encoding="utf-8")
            metadata = parse_optical_metadata(info)
            extend = read_extend_px(info)
        self.assertEqual(metadata.row_gsd_m, 1.08)
        self.assertEqual(metadata.column_gsd_m, 1.17)
        self.assertEqual(extend, 40)

    def test_volume_matches_legacy_height_ratio_formula(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 10, "test", 1)
        shadow = ShadowGeometry(12, 4, "test", True, "ok", 1, 1)
        result = calculate_volume(circle, shadow, metadata)
        self.assertTrue(result["geometry_valid"])
        self.assertAlmostEqual(result["tank_height_m"], 12.0, places=5)
        self.assertAlmostEqual(result["roof_depth_m"], 4.0, places=5)
        self.assertAlmostEqual(result["V_m3"], math.pi * 100 * 8, places=4)

    def test_small_inner_shadow_excess_is_reported_as_empty_tank(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 10, "test", 1)
        shadow = ShadowGeometry(12, 14, "test", True, "ok", 20, 20)
        result = calculate_volume(circle, shadow, metadata)
        self.assertTrue(result["geometry_valid"])
        self.assertEqual(result["geometry_status"], "oil_empty")
        self.assertEqual(result["oil_storage_ratio"], 0.0)
        self.assertEqual(result["V_m3"], 0.0)
        self.assertAlmostEqual(result["max_volume_m3"], math.pi * 100 * 12, places=4)

    def test_large_inner_shadow_excess_is_invalid_not_empty(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 10, "test", 1)
        shadow = ShadowGeometry(12, 16, "test", True, "ok", 20, 20)
        result = calculate_volume(circle, shadow, metadata)
        self.assertFalse(result["geometry_valid"])
        self.assertEqual(result["geometry_status"], "inner_shadow_exceeds_outer")
        self.assertTrue(math.isnan(result["V_m3"]))

    def test_conflicting_boundary_is_not_replaced_by_unrelated_arc(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 20, "test", 1)
        boundary = ShadowGeometry(
            12, 16, "legacy_boundary_scan", True, "ok", 30, 30
        )
        arc = ShadowGeometry(
            15, 5, "legacy_arc_match", True, "ok", 30, 30, 40, 40
        )
        selected = select_shadow_candidate(boundary, arc, circle, metadata)
        self.assertFalse(selected.valid)
        self.assertEqual(selected.status, "shadow_not_plausible")

    def test_implausible_tank_height_is_not_converted_to_volume(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 10, "test", 1)
        shadow = ShadowGeometry(80, 5, "test", True, "ok", 20, 20)
        result = calculate_volume(circle, shadow, metadata)
        self.assertFalse(result["geometry_valid"])
        self.assertEqual(result["geometry_status"], "implausible_tank_height")
        self.assertTrue(math.isnan(result["V_m3"]))

    def test_implausible_boundary_uses_plausible_arc_fallback(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 20, "test", 1)
        boundary = ShadowGeometry(
            80, 5, "legacy_boundary_scan", True, "ok", 30, 30
        )
        arc = ShadowGeometry(
            15, 5, "legacy_arc_match", True, "ok", 30, 30, 40, 40
        )
        selected = select_shadow_candidate(boundary, arc, circle, metadata)
        self.assertTrue(selected.valid)
        self.assertEqual(selected.method, "legacy_arc_match")
        self.assertIn("tank_height_above_max", selected.selection_reason)

    def test_recovery_arc_does_not_replace_implausible_boundary(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 20, "test", 1)
        boundary = ShadowGeometry(
            80, 5, "legacy_boundary_scan", True, "ok", 30, 30
        )
        recovery = ShadowGeometry(
            15, 5, "physical_arc_recovery", True, "ok", 30, 30, 40, 40
        )
        selected = select_shadow_candidate(boundary, recovery, circle, metadata)
        self.assertFalse(selected.valid)
        self.assertEqual(selected.status, "shadow_not_plausible")

    def test_weak_boundary_uses_arc_fallback(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 30, "test", 1)
        boundary = ShadowGeometry(
            15, 5, "legacy_boundary_scan", True, "ok", 3, 3
        )
        arc = ShadowGeometry(
            14, 5, "legacy_arc_match", True, "ok", 3, 3, 35, 35
        )
        selected = select_shadow_candidate(boundary, arc, circle, metadata)
        self.assertTrue(selected.valid)
        self.assertEqual(selected.method, "legacy_arc_match")
        self.assertIn("edge_support", selected.selection_reason)

    def test_plausible_weak_boundary_survives_when_arc_is_invalid(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 30, "test", 1)
        boundary = ShadowGeometry(
            15, 5, "legacy_boundary_scan", True, "ok", 3, 3
        )
        arc = ShadowGeometry(
            80, 5, "legacy_arc_match", True, "ok", 3, 3, 35, 35
        )
        selected = select_shadow_candidate(boundary, arc, circle, metadata)
        self.assertTrue(selected.valid)
        self.assertEqual(selected.method, "legacy_boundary_scan")
        self.assertEqual(selected.status, "ok_low_confidence")

    def test_arc_inner_completes_supported_boundary_outer(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 20, "test", 1)
        boundary = ShadowGeometry(
            15, 0, "legacy_boundary_scan", True, "ok", 30, 0
        )
        arc = ShadowGeometry(
            14, 6, "legacy_arc_match", True, "ok", 30, 0, 40, 60
        )
        selected = select_shadow_candidate(boundary, arc, circle, metadata)
        self.assertTrue(selected.valid)
        self.assertEqual(selected.method, "boundary_with_arc_inner")
        self.assertEqual(selected.lex_px, 15)
        self.assertEqual(selected.lin_px, 6)

    def test_boundary_inner_completes_arc_outer(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 20, "test", 1)
        boundary = ShadowGeometry(
            80, 6, "legacy_boundary_scan", True, "ok", 30, 30
        )
        arc = ShadowGeometry(
            15, 0, "legacy_arc_match", True, "ok", 30, 30, 40, 0
        )
        selected = select_shadow_candidate(boundary, arc, circle, metadata)
        self.assertTrue(selected.valid)
        self.assertEqual(selected.method, "arc_with_boundary_inner")
        self.assertEqual(selected.lex_px, 15)
        self.assertEqual(selected.lin_px, 6)

    def test_unconfirmed_full_tank_remains_valid_but_is_flagged(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        circle = CircleGeometry(50, 50, 20, "test", 1)
        boundary = ShadowGeometry(
            15, 0, "legacy_boundary_scan", True, "ok", 30, 0
        )
        arc = ShadowGeometry(
            15, 0, "legacy_arc_match", True, "ok", 30, 0, 40, 0
        )
        selected = select_shadow_candidate(boundary, arc, circle, metadata)
        result = calculate_volume(circle, selected, metadata)
        self.assertTrue(result["geometry_valid"])
        self.assertEqual(selected.status, "full_uncertain")
        self.assertEqual(result["geometry_status"], "full_uncertain")
        self.assertEqual(result["oil_storage_ratio"], 1.0)

    def test_height_bounds_are_converted_to_shadow_pixels(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 2.0, 2.0)
        self.assertEqual(tank_height_shadow_bounds_px(metadata, 8, 25), (4, 12))

    def test_plausibility_limits_are_configurable(self) -> None:
        metadata = OpticalMetadata(45, 0, 90, 0, 1.0, 1.0)
        shadow = ShadowGeometry(27, 5, "test", True, "ok", 20, 20)
        self.assertEqual(
            shadow_plausibility_reason(shadow, metadata),
            "tank_height_above_max",
        )
        self.assertIsNone(
            shadow_plausibility_reason(
                shadow,
                metadata,
                min_tank_height_m=8,
                max_tank_height_m=30,
            )
        )


if __name__ == "__main__":
    unittest.main()
