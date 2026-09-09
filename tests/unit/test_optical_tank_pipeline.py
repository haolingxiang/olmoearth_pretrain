"""Unit tests for the optical tank dataset adapter and physical model."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from scripts.tools.optical_tank_geometry import (
    CircleGeometry,
    OpticalMetadata,
    ShadowGeometry,
    calculate_volume,
    index_tank_pairs,
    parse_optical_metadata,
    read_extend_px,
    split_shadow_name,
)


class OpticalTankPipelineTests(unittest.TestCase):
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
        shadow = ShadowGeometry(12, 14, "test", True, "ok", 1, 1)
        result = calculate_volume(circle, shadow, metadata)
        self.assertTrue(result["geometry_valid"])
        self.assertEqual(result["geometry_status"], "oil_empty")
        self.assertEqual(result["oil_storage_ratio"], 0.0)
        self.assertEqual(result["V_m3"], 0.0)
        self.assertAlmostEqual(result["max_volume_m3"], math.pi * 100 * 12, places=4)


if __name__ == "__main__":
    unittest.main()
