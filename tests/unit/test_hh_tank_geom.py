"""CPU regression tests; no OlmoEarth downloads or GPU required.

Run: python -m unittest discover -s tests/unit -p test_hh_tank_geom.py -v
"""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

PATH = Path(__file__).resolve().parents[2] / "scripts/tools/train_hh_tank_geom.py"
SPEC = importlib.util.spec_from_file_location("tank_geom", PATH)
geom = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geom)


class FakeBackbone(nn.Module):
    """Differentiable substitute only for the unavailable pretrained encoder."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def encoder(self, sample, fast_pass, patch_size):
        x = sample.sentinel1[:, ::patch_size, ::patch_size]
        x = x.repeat(1, 1, 1, 1, 4).unsqueeze(4) * self.weight
        return {"tokens_and_masks": SimpleNamespace(sentinel1=x)}


class TankGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_paper_height_and_invalid_volume(self):
        points = [np.array([0.0, 0.0]), np.array([30.0, 0.0]), np.array([50.0, 0.0])]
        # radius=20: roof descent=10*Sr/cos(delta), oil depth=20*Sr/cos(delta).
        result = geom.volume_from_geometry(*points, 20.0, 0.5, 45.0)
        self.assertTrue(result["geometry_valid"])
        self.assertAlmostEqual(result["h_m"], 5 / math.cos(math.pi / 4))
        self.assertAlmostEqual(result["oil_height_raw_m"], 10 / math.cos(math.pi / 4))
        self.assertAlmostEqual(
            result["V_m3"], math.pi * result["R_m"] ** 2 * result["oil_height_raw_m"]
        )
        invalid = geom.volume_from_geometry(*points, 40.0, 0.5, 45.0)
        self.assertFalse(invalid["geometry_valid"])
        self.assertTrue(math.isnan(invalid["V_m3"]))
        legacy = geom.volume_from_geometry(*points, 60.0, 0.5, 45.0, formula="legacy")
        self.assertEqual(legacy["V_m3"], 0.0)

    def test_letterbox_preserves_geometry_and_ignores_padding(self):
        image = np.ones((80, 240), np.float32)
        mask, scale, ox, oy = geom.letterbox(image, 128, is_mask=True)
        points = np.array([[0.0, 0.0], [239.0, 79.0], [50.5, 30.25]])
        transformed = points * scale + [ox, oy]
        np.testing.assert_allclose((transformed - [ox, oy]) / scale, points, atol=1e-6)
        self.assertTrue(np.any(mask == -100))
        restored = geom.restore_mask(
            (mask == 1).astype(np.uint8), image.shape, "letterbox"
        )
        self.assertEqual(restored.shape, image.shape)
        self.assertGreater(restored.mean(), 0.95)

    def test_clean_circle_not_distracted_by_remote_speckles(self):
        yy, xx = np.indices((128, 128))
        roof = ((xx - 62) ** 2 + (yy - 60) ** 2 <= 25**2).astype(np.uint8)
        roof[1:4, 1:4] = 1
        roof[60, 62] = 0
        cleaned = geom.clean_mask(roof)
        self.assertEqual(cleaned[1, 1], 0)
        self.assertEqual(cleaned[60, 62], 1)
        self.assertAlmostEqual(geom.robust_circle_radius(roof), 25.0, delta=1.0)

    def test_peak_decode_does_not_average_distant_background(self):
        logits = torch.zeros(1, 3, 128, 128)
        logits[:, :, 12, 18] = 8
        local = geom.decode_points(logits, "local")
        self.assertLess(
            float(torch.linalg.vector_norm(local[0, 0] - torch.tensor([18.0, 12.0]))),
            0.1,
        )
        soft = geom.decode_points(logits, "soft")
        self.assertGreater(
            float(torch.linalg.vector_norm(soft[0, 0] - local[0, 0])), 10.0
        )

    def test_loss_finite_and_gradients_reach_projection(self):
        stub = SimpleNamespace(
            MaskedOlmoEarthSample=SimpleNamespace,
            MaskValue=SimpleNamespace(ONLINE_ENCODER=SimpleNamespace(value=0)),
        )
        with patch.dict("sys.modules", {"olmoearth_pretrain.datatypes": stub}):
            model = geom.HHTankGeomModel(
                FakeBackbone(), 8, mid_dim=64, detail_branch=True
            )
            model.train()
            out = model(torch.randn(2, 2, 32, 32))
            mask = torch.zeros(2, 32, 32, dtype=torch.long)
            mask[:, 8:24, 8:24] = 1
            mask[:, :2] = -100
            points = torch.tensor([[[8.0, 16.0], [16.0, 16.0], [24.0, 16.0]]] * 2)
            heats = np.stack(
                [
                    geom._gaussian_heatmap(32, 32, float(x), float(y))
                    for x, y in points[0]
                ]
            )
            heatmaps = torch.tensor(np.stack([heats] * 2))
            loss = geom.multitask_loss(out, mask, heatmaps, points=points)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.proj.weight.grad)
            self.assertGreater(float(model.proj.weight.grad.abs().sum()), 0.0)
            self.assertIsNone(model.backbone.weight.grad)
            self.assertIsNotNone(model.head.detail[0].weight.grad)
            self.assertFalse(model.backbone.training)

    def test_ignored_padding_does_not_change_segmentation_loss(self):
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        target[:, :4] = -100
        logits = torch.zeros(1, 2, 8, 8)
        changed = logits.clone()
        changed[:, 1, :4] = 100
        self.assertAlmostEqual(
            float(geom.dice_loss_with_logits(logits, target)),
            float(geom.dice_loss_with_logits(changed, target)),
        )

    def test_old_head_checkpoint_loads_without_new_branch(self):
        old = geom.HHTankGeomModel(FakeBackbone(), 8, mid_dim=64)
        new = geom.HHTankGeomModel(FakeBackbone(), 8, mid_dim=64, detail_branch=False)
        new.head.load_state_dict(old.head.state_dict(), strict=True)
        new.proj.load_state_dict(old.proj.state_dict(), strict=True)

    def test_missing_keypoint_has_no_heatmap_gradient(self):
        logits = torch.zeros(1, 3, 16, 16, requires_grad=True)
        heats = torch.zeros_like(logits)
        heats[0, 0, 5, 4] = 1
        heats[0, 1, 5, 9] = 1
        loss = geom.multitask_loss(
            {"seg_logits": torch.zeros(1, 2, 16, 16), "kp_heatmaps": logits},
            torch.zeros(1, 16, 16, dtype=torch.long),
            heats,
            points=torch.tensor([[[4.0, 5.0], [9.0, 5.0], [0.0, 0.0]]]),
            kp_valid=torch.tensor([[True, True, False]]),
        )
        loss.backward()
        self.assertEqual(float(logits.grad[0, 2].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[0, 0].abs().sum()), 0.0)

    def test_training_uses_full_train_and_test_validation(self):
        loaded_splits = []

        class TinyDataset(torch.utils.data.Dataset):
            preprocess = "letterbox"

            def __init__(self, root, split, size, preprocess="letterbox"):
                self.paths = [Path(f"tank{i}.tif") for i in range(4)]
                self.size = size
                self.split = split
                loaded_splits.append(split)

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, i):
                points = np.array([[5.0, 16.0], [15.0, 16.0], [25.0, 16.0]])
                heats = np.stack(
                    [geom._gaussian_heatmap(32, 32, x, y) for x, y in points]
                )
                yy, xx = np.indices((32, 32))
                return {
                    "image": torch.randn(2, 32, 32),
                    "mask": torch.tensor(
                        ((xx - 16) ** 2 + (yy - 16) ** 2 < 8**2).astype(np.int64)
                    ),
                    "heatmaps": torch.tensor(heats),
                    "points": torch.tensor(points).float(),
                    "kp_valid": torch.ones(3, dtype=torch.bool),
                    "r_mask_px": torch.tensor(8.0),
                    "scale": torch.tensor(1.0),
                    "offset": torch.tensor([0.0, 0.0]),
                    "original_shape": torch.tensor([32, 32]),
                    "name": self.paths[i].name,
                    "split": self.split,
                }

        stub = SimpleNamespace(
            MaskedOlmoEarthSample=SimpleNamespace,
            MaskValue=SimpleNamespace(ONLINE_ENCODER=SimpleNamespace(value=0)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = geom._build_parser().parse_args(
                [
                    "train",
                    "--data-root",
                    tmp,
                    "--weights",
                    tmp,
                    "--out-dir",
                    tmp,
                    "--size",
                    "32",
                    "--epochs",
                    "1",
                    "--workers",
                    "0",
                    "--mid-dim",
                    "64",
                ]
            )
            model = geom.HHTankGeomModel(
                FakeBackbone(), 8, mid_dim=64, detail_branch=True
            )
            with (
                patch.dict("sys.modules", {"olmoearth_pretrain.datatypes": stub}),
                patch.object(geom, "TankGeomDataset", TinyDataset),
                patch.object(geom, "build_model", return_value=(model, {"emb_dim": 8})),
            ):
                geom.train(args)
            checkpoint = torch.load(Path(tmp) / "best.pt", weights_only=False)
            self.assertEqual(checkpoint["preprocess"], "letterbox")
            manifest = checkpoint["split_manifest"]
            self.assertEqual(loaded_splits, ["train", "test"])
            self.assertEqual(manifest["validation_split"], "test")
            self.assertEqual(manifest["train"], [f"train/tank{i}.tif" for i in range(4)])
            self.assertEqual(manifest["val"], [f"test/tank{i}.tif" for i in range(4)])
            self.assertTrue((Path(tmp) / "history.json").exists())

            class MaskWriter:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def write(self, arr, band):
                    if arr.shape != (32, 32):
                        raise AssertionError(arr.shape)

            # Exercise export and JSON/CSV metadata with only external model
            # loading and raster IO replaced. The geometry/metrics are real.
            meta = Path(tmp) / "meta.csv"
            meta.write_text(
                "filename,pixel_resolution,incidenceangle\n"
                + "".join(f"tank{i}.tif,0.5,45\n" for i in range(4))
            )
            ev = geom._build_parser().parse_args(
                [
                    "eval",
                    "--data-root",
                    tmp,
                    "--weights",
                    tmp,
                    "--ckpt",
                    str(Path(tmp) / "best.pt"),
                    "--split",
                    "train",
                    "--workers",
                    "0",
                    "--meta",
                    str(meta),
                    "--save-preds",
                    str(Path(tmp) / "preds"),
                ]
            )
            model_loader = SimpleNamespace(
                load_model_from_path=lambda _: FakeBackbone()
            )
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "olmoearth_pretrain.datatypes": stub,
                        "olmoearth_pretrain.model_loader": model_loader,
                        "rasterio": SimpleNamespace(open=lambda *a, **k: MaskWriter()),
                    },
                ),
                patch.object(geom, "TankGeomDataset", TinyDataset),
            ):
                geom.eval_only(ev)
            metrics = json.loads((Path(tmp) / "preds/metrics.json").read_text())
            self.assertEqual(metrics["n_predictions"], 4)
            self.assertEqual(metrics["missing_meta"], 0)
            self.assertEqual(metrics["formula"], "paper")
            self.assertTrue((Path(tmp) / "preds/volumes.csv").exists())


if __name__ == "__main__":
    unittest.main()
