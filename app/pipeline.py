from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROOT = Path(__file__).resolve().parents[1]
SCRAPED_DIR = ROOT / "data" / "scraped"
SCRAPED_HISTORY_DIR = SCRAPED_DIR / "history"
PREDICTIONS_DIR = ROOT / "predictions"
PREDICTIONS_HISTORY_DIR = PREDICTIONS_DIR / "history"
MODELS_DIR = ROOT / "models" / "builded"
TOKEN_FILE = ROOT / ".confing"

OWNER = "gandpablo"
REPO = "VALENCIA_DATA_EDM"
BRANCH = "main"
LOCAL_TIMEZONE = "Europe/Madrid"
SCRAPER_URL = "https://www.valencia.es/valenciaalminut/"
SCRAPER_TABLE_ID = "tabla_dinamica"

POLLUTANTS = ["SO2", "NO2", "O3", "PM10", "PM2.5"]
SCRAPER_COLUMNS = ["µg/m3", "SO2", "NO2", "O3", "PM-10", "PM-2.5"]
SCRAPER_TO_MODEL = {"SO2": "SO2", "NO2": "NO2", "O3": "O3", "PM-10": "PM10", "PM-2.5": "PM2.5"}
MODEL_TO_SCRAPER = {value: key for key, value in SCRAPER_TO_MODEL.items()}

NAME_MAP = {
    "AVDA.FRANCIA": "València - Av. França",
    "BULEVARD SUD": "València - Bulevard Sud",
    "MOLÍ DEL SOL": "València - Molí del Sol",
    "PISTA DE SILLA": "València - Pista de Silla",
    "POLITÈCNIC": "València - Politècnic",
    "VIVERS": "València - Vivers",
    "VALÈNCIA CENTRE": "València - Centre",
    "OLIVERETA": "València Olivereta",
}


def ensure_dirs() -> None:
    for path in [SCRAPED_HISTORY_DIR, PREDICTIONS_HISTORY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def local_now() -> datetime:
    return datetime.now(ZoneInfo(LOCAL_TIMEZONE))


def timestamp_name() -> str:
    return local_now().strftime("%Y-%m-%d_%H-%M.csv")


def read_token() -> str:
    token = os.environ.get("EDM_GITHUB_TOKEN")
    if token:
        return token
    if not TOKEN_FILE.exists():
        raise RuntimeError("No GitHub token found. Set EDM_GITHUB_TOKEN or create .confing.")
    text = TOKEN_FILE.read_text(encoding="utf-8")
    match = re.search(r"EDM_GITHUB_TOKEN\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        raise RuntimeError("Could not read EDM_GITHUB_TOKEN from .confing")
    return match.group(1)


def github_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_url(path: str) -> str:
    return f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}"


def remote_file(token: str, path: str) -> dict | None:
    response = requests.get(
        github_url(path),
        headers=github_headers(token),
        params={"ref": BRANCH},
        timeout=30,
    )
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(f"Error checking {path}: {response.status_code} - {response.text}")
    return response.json()


def git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def upload_file(token: str, path: str, content: bytes, message: str) -> None:
    remote = remote_file(token, path)
    if remote and remote.get("sha") == git_blob_sha(content):
        return
    payload = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": BRANCH,
    }
    if remote:
        payload["sha"] = remote["sha"]

    response = requests.put(
        github_url(path),
        headers=github_headers(token),
        json=payload,
        timeout=60,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Error uploading {path}: {response.status_code} - {response.text}")


def scrape_current_table() -> pd.DataFrame:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-notifications")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--window-size=1920,1080")

    chrome_binary = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if chrome_binary:
        options.binary_location = chrome_binary
    chromedriver = shutil.which("chromedriver")

    driver = (
        webdriver.Chrome(service=Service(chromedriver), options=options)
        if chromedriver
        else webdriver.Chrome(options=options)
    )
    try:
        driver.get(SCRAPER_URL)
        table_html = WebDriverWait(driver, 40).until(
            EC.presence_of_element_located((By.ID, SCRAPER_TABLE_ID))
        ).get_attribute("outerHTML")
    finally:
        driver.quit()

    df = pd.read_html(StringIO(table_html))[0]
    df = df[SCRAPER_COLUMNS].dropna(how="all")
    if df.empty or df["µg/m3"].dropna().empty:
        raise RuntimeError("Scrape returned no station rows; keeping previous data.")
    return df


def normalized_scrape(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SCRAPER_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Scrape missing required columns: {', '.join(missing)}")

    normalized = df[SCRAPER_COLUMNS].copy()
    normalized["µg/m3"] = normalized["µg/m3"].fillna("").astype(str).str.strip()
    for column in SCRAPER_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(float)
    return normalized.sort_values("µg/m3").reset_index(drop=True)


def scrape_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        left_normalized = normalized_scrape(left)
        right_normalized = normalized_scrape(right)
    except ValueError:
        return False
    if left_normalized.shape != right_normalized.shape:
        return False
    if not left_normalized["µg/m3"].equals(right_normalized["µg/m3"]):
        return False
    return bool(
        np.allclose(
            left_normalized[SCRAPER_COLUMNS[1:]].to_numpy(dtype=float),
            right_normalized[SCRAPER_COLUMNS[1:]].to_numpy(dtype=float),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    )


def collapse_consecutive_values(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if value_column not in frame.columns:
        raise ValueError(f"Missing value column: {value_column}")

    values = pd.to_numeric(frame[value_column], errors="coerce")
    previous = values.shift()
    unchanged = values.eq(previous) | (values.isna() & previous.isna())
    return frame.loc[~unchanged].copy()


def latest_history_scrape() -> tuple[Path, pd.DataFrame] | None:
    latest_distinct: tuple[Path, pd.DataFrame] | None = None
    for path in sorted(SCRAPED_HISTORY_DIR.glob("*.csv")):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        if latest_distinct is None or not scrape_frames_equal(frame, latest_distinct[1]):
            latest_distinct = path, frame
    return latest_distinct


def save_scrape(df: pd.DataFrame, filename: str) -> tuple[Path, bool]:
    ensure_dirs()
    if df.empty or df["µg/m3"].dropna().empty:
        raise RuntimeError("Refusing to save an empty scrape.")
    normalized_scrape(df)
    latest_path = SCRAPED_DIR / "latest.csv"
    df.to_csv(latest_path, index=False, encoding="utf-8-sig")

    previous = latest_history_scrape()
    if previous is not None and scrape_frames_equal(df, previous[1]):
        return previous[0], False

    history_path = SCRAPED_HISTORY_DIR / filename
    df.to_csv(history_path, index=False, encoding="utf-8-sig")
    update_index(SCRAPED_DIR / "index.json", "data/scraped/history", SCRAPED_HISTORY_DIR)
    return history_path, True


def update_index(index_path: Path, remote_prefix: str, folder: Path) -> None:
    items = [
        {"path": f"{remote_prefix}/{path.name}", "timestamp": path.stem}
        for path in sorted(folder.glob("*.csv"))
    ]
    index_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def decode_remote_content(remote: dict, path: str) -> bytes:
    content = remote.get("content")
    if remote.get("encoding") == "base64" and isinstance(content, str):
        return base64.b64decode(content)
    download_url = remote.get("download_url")
    if not download_url:
        raise RuntimeError(f"GitHub did not provide downloadable content for {path}.")
    response = requests.get(download_url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Error downloading {path}: {response.status_code} - {response.text}")
    return response.content


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def sync_indexed_history(
    token: str,
    index_remote_path: str,
    remote_prefix: str,
    local_index_path: Path,
    local_folder: Path,
) -> int:
    remote_index = remote_file(token, index_remote_path)
    if remote_index is None:
        return 0
    index_bytes = decode_remote_content(remote_index, index_remote_path)
    items = json.loads(index_bytes.decode("utf-8-sig"))
    downloaded = 0
    for item in items:
        remote_path = str(item.get("path", ""))
        if not remote_path.startswith(f"{remote_prefix}/") or not remote_path.endswith(".csv"):
            continue
        filename = Path(remote_path).name
        local_path = local_folder / filename
        if local_path.exists():
            continue
        remote = remote_file(token, remote_path)
        if remote is None:
            continue
        write_bytes_atomic(local_path, decode_remote_content(remote, remote_path))
        downloaded += 1
    write_bytes_atomic(local_index_path, index_bytes)
    return downloaded


def sync_remote_data() -> dict[str, int]:
    ensure_dirs()
    token = read_token()
    downloaded_scrapes = sync_indexed_history(
        token,
        "data/scraped/index.json",
        "data/scraped/history",
        SCRAPED_DIR / "index.json",
        SCRAPED_HISTORY_DIR,
    )
    downloaded_predictions = sync_indexed_history(
        token,
        "predictions/index.json",
        "predictions/history",
        PREDICTIONS_DIR / "index.json",
        PREDICTIONS_HISTORY_DIR,
    )
    for remote_path, local_path in [
        ("data/scraped/latest.csv", SCRAPED_DIR / "latest.csv"),
        ("predictions/latest.csv", PREDICTIONS_DIR / "latest.csv"),
    ]:
        remote = remote_file(token, remote_path)
        if remote is not None:
            write_bytes_atomic(local_path, decode_remote_content(remote, remote_path))
    return {
        "downloaded_scrapes": downloaded_scrapes,
        "downloaded_predictions": downloaded_predictions,
    }


def load_scrape(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["station_model"] = df["µg/m3"].map(NAME_MAP).fillna(df["µg/m3"])
    for col in SCRAPER_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_scraped_history(current_path: Path) -> pd.DataFrame:
    rows = []
    previous_snapshot: pd.DataFrame | None = None
    for path in sorted(SCRAPED_HISTORY_DIR.glob("*.csv")):
        ts = pd.to_datetime(path.stem, format="%Y-%m-%d_%H-%M", errors="coerce")
        if pd.isna(ts):
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        if previous_snapshot is not None and scrape_frames_equal(df, previous_snapshot):
            continue
        previous_snapshot = df
        df["timestamp"] = ts
        rows.append(df)

    if not rows:
        raise RuntimeError("No scraped history available to build prediction features.")
    data = pd.concat(rows, ignore_index=True)
    data["station_model"] = data["µg/m3"].map(NAME_MAP).fillna(data["µg/m3"])
    for col in SCRAPER_COLUMNS[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["timestamp"]).sort_values("timestamp")


def load_model(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def feature_row(history: pd.DataFrame, station: str, model: dict) -> pd.DataFrame | None:
    station_history = history[history["station_model"] == station].sort_values("timestamp")
    if station_history.empty:
        return None

    latest = station_history.iloc[-1]
    current_time = latest["timestamp"]
    features: dict[str, float] = {}
    needed = model["feature_names"]

    for pollutant in POLLUTANTS:
        scraper_col = MODEL_TO_SCRAPER[pollutant]
        features[f"{pollutant}_current"] = float(latest.get(scraper_col, np.nan))

    target = model["pollutant"]
    target_col = MODEL_TO_SCRAPER[target]
    current_target = float(latest.get(target_col, np.nan))
    if pd.isna(current_target):
        return None

    lag_pattern = re.compile(rf"^{re.escape(target)}_lag_(\d+)h$")
    lags = [int(match.group(1)) for name in needed if (match := lag_pattern.match(name))]
    rolling_hours = []
    if f"{target}_rolling_24h" in needed:
        rolling_hours.append(24)
    if f"{target}_rolling_7d" in needed:
        rolling_hours.append(168)
    required_hours = max([*lags, *rolling_hours], default=0)

    target_series = (
        station_history.dropna(subset=[target_col])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .set_index("timestamp")[target_col]
        .sort_index()
        .astype(float)
    )
    if target_series.empty:
        return None

    hourly_index = pd.date_range(
        end=current_time,
        periods=required_hours + 1,
        freq="h",
    )
    interpolation_index = target_series.index.union(hourly_index)
    hourly_target = (
        target_series.reindex(interpolation_index)
        .sort_index()
        .interpolate(method="time", limit_area="inside")
        .reindex(hourly_index)
    )
    if hourly_target.isna().any():
        return None

    for lag in lags:
        features[f"{target}_lag_{lag}h"] = float(hourly_target.loc[current_time - pd.Timedelta(hours=lag)])

    if f"{target}_rolling_24h" in needed:
        features[f"{target}_rolling_24h"] = float(hourly_target.tail(24).mean())
    if f"{target}_rolling_7d" in needed:
        features[f"{target}_rolling_7d"] = float(hourly_target.tail(168).mean())

    hour = current_time.hour
    dow = current_time.dayofweek
    month = current_time.month
    features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    features["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    features["dow_cos"] = math.cos(2 * math.pi * dow / 7)
    features["month_sin"] = math.sin(2 * math.pi * month / 12)
    features["month_cos"] = math.cos(2 * math.pi * month / 12)

    row = pd.DataFrame([features])
    if row[needed].isna().any(axis=None):
        return None
    return row


def row_outside_training_bounds(model: dict, row: pd.DataFrame) -> bool:
    if "feature_lower_bounds" not in model or "feature_upper_bounds" not in model:
        return False
    cols = model["feature_names"]
    lower = pd.Series(model["feature_lower_bounds"])[cols]
    upper = pd.Series(model["feature_upper_bounds"])[cols]
    return bool(((row[cols] < lower) | (row[cols] > upper)).any(axis=None))


def predict_model(model: dict, row: pd.DataFrame) -> float:
    cols = model["feature_names"]
    means = pd.Series(model["feature_means"])[cols]
    stds = pd.Series(model["feature_stds"])[cols]
    values = row[cols]
    if "feature_lower_bounds" in model and "feature_upper_bounds" in model:
        lower = pd.Series(model["feature_lower_bounds"])[cols]
        upper = pd.Series(model["feature_upper_bounds"])[cols]
        values = values.clip(lower=lower, upper=upper, axis=1)
    scaled = ((values - means) / stds).to_numpy(dtype=float)
    prediction = model["intercept"] + scaled @ np.array(model["coefficients"], dtype=float)
    return float(np.asarray(prediction).item())


def make_predictions(current_path: Path, filename: str) -> tuple[Path, dict[str, int]]:
    ensure_dirs()
    current = load_scrape(current_path)
    history = load_scraped_history(current_path)
    registry = json.loads((MODELS_DIR / "registry.json").read_text(encoding="utf-8"))

    predictions = current[["µg/m3"]].copy()
    for col in SCRAPER_COLUMNS[1:]:
        predictions[col] = np.nan

    stats = {
        "expected_predictions": 0,
        "model_predictions": 0,
        "baseline_predictions": 0,
        "baseline_selected_strategy": 0,
        "baseline_insufficient_history": 0,
        "clipped_feature_rows": 0,
    }
    seen: set[tuple[str, str]] = set()

    for entry in registry:
        if int(entry.get("horizon_hours", 0)) != 8:
            continue
        model = load_model(entry["model_path"])
        station = model["station"]
        pollutant = model["pollutant"]
        model_key = (station, pollutant)
        if model_key in seen:
            raise RuntimeError(f"Duplicate +8h model in registry: {station} / {pollutant}")
        seen.add(model_key)
        scraper_col = MODEL_TO_SCRAPER[pollutant]
        mask = current["station_model"] == station
        if int(mask.sum()) > 1:
            raise RuntimeError(f"Duplicate station in current scrape: {station}")
        if not mask.any():
            continue
        current_value = current.loc[mask, scraper_col].iloc[0]
        if pd.isna(current_value):
            continue

        stats["expected_predictions"] += 1
        row = feature_row(history, station, model)
        selected_strategy = model.get(
            "selected_strategy",
            "ridge" if bool(model.get("beats_baseline_mae", False)) else "persistence",
        )
        if selected_strategy != "ridge":
            value = float(current_value)
            stats["baseline_predictions"] += 1
            stats["baseline_selected_strategy"] += 1
        elif row is None:
            value = float(current_value)
            stats["baseline_predictions"] += 1
            stats["baseline_insufficient_history"] += 1
        else:
            if row_outside_training_bounds(model, row):
                stats["clipped_feature_rows"] += 1
            value = max(0.0, predict_model(model, row))
            stats["model_predictions"] += 1
        predictions.loc[mask, scraper_col] = round(value, 1)

    generated = int(predictions[SCRAPER_COLUMNS[1:]].count().sum())
    if generated != stats["expected_predictions"]:
        raise RuntimeError(
            f"Incomplete prediction output: generated {generated} of {stats['expected_predictions']} expected values."
        )
    numeric_values = predictions[SCRAPER_COLUMNS[1:]].stack().dropna().to_numpy(dtype=float)
    if generated == 0:
        raise RuntimeError("Prediction output is empty.")
    if not np.isfinite(numeric_values).all() or (numeric_values < 0).any():
        raise RuntimeError("Prediction output contains invalid values.")

    history_path = PREDICTIONS_HISTORY_DIR / filename
    latest_path = PREDICTIONS_DIR / "latest.csv"
    predictions.to_csv(history_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(latest_path, index=False, encoding="utf-8-sig")
    update_index(PREDICTIONS_DIR / "index.json", "predictions/history", PREDICTIONS_HISTORY_DIR)
    return history_path, stats


def upload_outputs(scrape_path: Path, prediction_path: Path) -> None:
    token = read_token()
    files = [
        ("data/scraped/latest.csv", SCRAPED_DIR / "latest.csv"),
        (f"data/scraped/history/{scrape_path.name}", scrape_path),
        ("data/scraped/index.json", SCRAPED_DIR / "index.json"),
        ("predictions/latest.csv", PREDICTIONS_DIR / "latest.csv"),
        (f"predictions/history/{prediction_path.name}", prediction_path),
        ("predictions/index.json", PREDICTIONS_DIR / "index.json"),
    ]
    for remote_path, local_path in files:
        upload_file(token, remote_path, local_path.read_bytes(), f"Update {remote_path}")


def run_manual_pipeline() -> dict[str, object]:
    ensure_dirs()
    sync_stats = sync_remote_data()
    filename = timestamp_name()
    df = scrape_current_table()
    scrape_path, snapshot_changed = save_scrape(df, filename)
    prediction_filename = filename if snapshot_changed else scrape_path.name
    prediction_path, prediction_stats = make_predictions(SCRAPED_DIR / "latest.csv", prediction_filename)
    upload_outputs(scrape_path, prediction_path)
    return {
        "scrape_file": scrape_path.name,
        "prediction_file": prediction_path.name,
        "scrape_path": str(scrape_path),
        "prediction_path": str(prediction_path),
        "snapshot_changed": snapshot_changed,
        "history_updated": snapshot_changed,
        "predictions_updated": True,
        **sync_stats,
        **prediction_stats,
    }
