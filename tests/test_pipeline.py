from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from base64 import b64encode
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import pipeline


def sample_scrape(no2_first_station: float = 20.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "µg/m3": ["ESTACION A", "ESTACION B"],
            "SO2": [4.0, np.nan],
            "NO2": [no2_first_station, 31.0],
            "O3": [55.0, 42.0],
            "PM-10": [18.0, 22.0],
            "PM-2.5": [9.0, 11.0],
        }
    )


class PipelineAuditTests(unittest.TestCase):
    def test_remote_history_sync_downloads_only_missing_files(self) -> None:
        index = [
            {"path": "data/scraped/history/first.csv", "timestamp": "first"},
            {"path": "data/scraped/history/second.csv", "timestamp": "second"},
        ]
        remote_files = {
            "data/scraped/index.json": json.dumps(index).encode(),
            "data/scraped/history/first.csv": b"first remote",
            "data/scraped/history/second.csv": b"second remote",
        }

        def fake_remote_file(_token: str, path: str) -> dict | None:
            content = remote_files.get(path)
            if content is None:
                return None
            return {
                "encoding": "base64",
                "content": b64encode(content).decode(),
            }

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "history"
            folder.mkdir()
            (folder / "first.csv").write_bytes(b"existing local")
            index_path = Path(tmp) / "index.json"
            with patch("pipeline.remote_file", side_effect=fake_remote_file):
                downloaded = pipeline.sync_indexed_history(
                    "token",
                    "data/scraped/index.json",
                    "data/scraped/history",
                    index_path,
                    folder,
                )

            self.assertEqual(downloaded, 1)
            self.assertEqual((folder / "first.csv").read_bytes(), b"existing local")
            self.assertEqual((folder / "second.csv").read_bytes(), b"second remote")
            self.assertEqual(json.loads(index_path.read_text()), index)

    def test_consecutive_equal_values_are_grouped_per_series(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-06-23 10:00", periods=5, freq="h"),
                "O3": [55.0, 55.0, 56.0, 56.0, 55.0],
            }
        )

        grouped = pipeline.collapse_consecutive_values(frame, "O3")

        self.assertEqual(grouped["O3"].tolist(), [55.0, 56.0, 55.0])
        self.assertEqual(grouped["timestamp"].dt.strftime("%H:%M").tolist(), ["10:00", "12:00", "14:00"])

    def test_identical_scrape_is_not_saved_twice(self) -> None:
        original_scraped_dir = pipeline.SCRAPED_DIR
        original_history_dir = pipeline.SCRAPED_HISTORY_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pipeline.SCRAPED_DIR = Path(tmp) / "scraped"
                pipeline.SCRAPED_HISTORY_DIR = pipeline.SCRAPED_DIR / "history"

                first_path, first_created = pipeline.save_scrape(
                    sample_scrape(),
                    "2026-06-23_10-00.csv",
                )
                duplicate_path, duplicate_created = pipeline.save_scrape(
                    sample_scrape().iloc[::-1],
                    "2026-06-23_10-01.csv",
                )
                changed_path, changed_created = pipeline.save_scrape(
                    sample_scrape(no2_first_station=21.0),
                    "2026-06-23_10-02.csv",
                )

                self.assertTrue(first_created)
                self.assertFalse(duplicate_created)
                self.assertEqual(duplicate_path, first_path)
                self.assertTrue(changed_created)
                self.assertNotEqual(changed_path, first_path)
                self.assertEqual(len(list(pipeline.SCRAPED_HISTORY_DIR.glob("*.csv"))), 2)
                index = json.loads((pipeline.SCRAPED_DIR / "index.json").read_text(encoding="utf-8"))
                self.assertEqual(len(index), 2)
        finally:
            pipeline.SCRAPED_DIR = original_scraped_dir
            pipeline.SCRAPED_HISTORY_DIR = original_history_dir

    def test_history_loader_collapses_only_consecutive_duplicate_snapshots(self) -> None:
        original_scraped_dir = pipeline.SCRAPED_DIR
        original_history_dir = pipeline.SCRAPED_HISTORY_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pipeline.SCRAPED_DIR = Path(tmp) / "scraped"
                pipeline.SCRAPED_HISTORY_DIR = pipeline.SCRAPED_DIR / "history"
                pipeline.SCRAPED_HISTORY_DIR.mkdir(parents=True)

                first = sample_scrape()
                changed = sample_scrape(no2_first_station=21.0)
                snapshots = [first, first.iloc[::-1], changed, first]
                names = [
                    "2026-06-23_10-00.csv",
                    "2026-06-23_10-01.csv",
                    "2026-06-23_10-02.csv",
                    "2026-06-23_10-03.csv",
                ]
                for name, snapshot in zip(names, snapshots):
                    snapshot.to_csv(pipeline.SCRAPED_HISTORY_DIR / name, index=False, encoding="utf-8-sig")

                history = pipeline.load_scraped_history(pipeline.SCRAPED_DIR / "latest.csv")
                timestamps = history["timestamp"].drop_duplicates().dt.strftime("%H:%M").tolist()

                self.assertEqual(timestamps, ["10:00", "10:02", "10:03"])
        finally:
            pipeline.SCRAPED_DIR = original_scraped_dir
            pipeline.SCRAPED_HISTORY_DIR = original_history_dir

    def test_model_registry_and_artifacts_are_consistent(self) -> None:
        registry = json.loads((pipeline.MODELS_DIR / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(len(registry), 64)
        self.assertEqual(sum(int(entry["horizon_hours"]) == 8 for entry in registry), 32)

        keys: set[tuple[str, str, int]] = set()
        for entry in registry:
            key = (entry["station"], entry["pollutant"], int(entry["horizon_hours"]))
            self.assertNotIn(key, keys)
            keys.add(key)

            model = pipeline.load_model(entry["model_path"])
            feature_names = model["feature_names"]
            self.assertIn(model["selected_strategy"], {"ridge", "persistence"})
            self.assertIn("validation_metrics", model)
            self.assertIn("validation_baseline_metrics", model)
            self.assertEqual(len(feature_names), len(model["coefficients"]))
            self.assertEqual(set(feature_names), set(model["feature_means"]))
            self.assertEqual(set(feature_names), set(model["feature_stds"]))
            self.assertEqual(set(feature_names), set(model["feature_lower_bounds"]))
            self.assertEqual(set(feature_names), set(model["feature_upper_bounds"]))

            numeric_values = [
                model["intercept"],
                *model["coefficients"],
                *model["feature_means"].values(),
                *model["feature_stds"].values(),
                *model["feature_lower_bounds"].values(),
                *model["feature_upper_bounds"].values(),
            ]
            self.assertTrue(all(math.isfinite(float(value)) for value in numeric_values))
            self.assertTrue(all(float(value) > 0 for value in model["feature_stds"].values()))
            self.assertTrue(
                all(
                    model["feature_lower_bounds"][name] <= model["feature_upper_bounds"][name]
                    for name in feature_names
                )
            )

    def test_live_prediction_output_is_complete_and_non_empty(self) -> None:
        original_predictions_dir = pipeline.PREDICTIONS_DIR
        original_history_dir = pipeline.PREDICTIONS_HISTORY_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pipeline.PREDICTIONS_DIR = Path(tmp) / "predictions"
                pipeline.PREDICTIONS_HISTORY_DIR = pipeline.PREDICTIONS_DIR / "history"
                output_path, stats = pipeline.make_predictions(
                    pipeline.SCRAPED_DIR / "latest.csv",
                    "audit.csv",
                )
                output = pd.read_csv(output_path, encoding="utf-8-sig")
        finally:
            pipeline.PREDICTIONS_DIR = original_predictions_dir
            pipeline.PREDICTIONS_HISTORY_DIR = original_history_dir

        values = output[pipeline.SCRAPER_COLUMNS[1:]].stack().dropna().to_numpy(dtype=float)
        self.assertGreater(stats["model_predictions"], 0)
        self.assertEqual(
            stats["model_predictions"] + stats["baseline_predictions"],
            stats["expected_predictions"],
        )
        self.assertEqual(len(values), stats["expected_predictions"])
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all())

    def test_insufficient_history_is_not_fabricated(self) -> None:
        registry = json.loads((pipeline.MODELS_DIR / "registry.json").read_text(encoding="utf-8"))
        model_entry = next(
            entry
            for entry in registry
            if int(entry["horizon_hours"]) == 8 and entry["beats_baseline_mae"]
        )
        model = pipeline.load_model(model_entry["model_path"])
        history = pipeline.load_scraped_history(pipeline.SCRAPED_DIR / "latest.csv")
        station_history = history[history["station_model"] == model["station"]]
        cutoff = station_history["timestamp"].max() - pd.Timedelta(hours=24)
        short_history = history[history["timestamp"] >= cutoff]
        self.assertIsNone(pipeline.feature_row(short_history, model["station"], model))


if __name__ == "__main__":
    unittest.main()
