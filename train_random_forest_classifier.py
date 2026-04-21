from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None


FEATURE_COLS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "dist_sma_10",
    "dist_sma_20",
    "zscore_20",
    "atr_14",
]

SELL_CLASS = -1
FLAT_CLASS = 0
BUY_CLASS = 1
CLASS_ORDER = [SELL_CLASS, FLAT_CLASS, BUY_CLASS]


def fetch_rates_from_mt5(symbol: str, timeframe_name: str, bars: int) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError(
            "The MetaTrader5 package for Python is not installed. Install it with: pip install MetaTrader5"
        )

    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if timeframe_name not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe_name}")

    if not mt5.initialize():
        raise RuntimeError(f"initialize() failed: {mt5.last_error()}")

    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe_name], 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"Could not read data for {symbol} {timeframe_name}. last_error={mt5.last_error()}"
            )
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[["time", "open", "high", "low", "close", "volume"]].copy()
    finally:
        mt5.shutdown()



def load_rates_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"time", "open", "high", "low", "close"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV does not contain mandatory columns: {sorted(missing)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    return df[["time", "open", "high", "low", "close", "volume"]]



def build_features(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["dist_sma_10"] = (df["close"] / df["sma_10"]) - 1.0
    df["dist_sma_20"] = (df["close"] / df["sma_20"]) - 1.0

    roll_mean_20 = df["close"].rolling(20).mean()
    roll_std_20 = df["close"].rolling(20).std()
    df["zscore_20"] = (df["close"] - roll_mean_20) / roll_std_20

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()

    # Target aligned with the strategy horizon: future return over N closed bars.
    df["fwd_ret_h"] = df["close"].shift(-horizon_bars) / df["close"] - 1.0

    df = df.dropna(subset=FEATURE_COLS + ["fwd_ret_h"]).copy()
    return df



def split_train_test(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_ratio < 0.95:
        raise ValueError("train_ratio must be between 0.50 and 0.95")
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if len(train_df) < 1000 or len(test_df) < 200:
        raise ValueError("Too few examples after split. Increase the number of bars or adjust train_ratio.")
    return train_df, test_df



def compute_return_barrier(train_df: pd.DataFrame, label_quantile: float) -> float:
    barrier = float(train_df["fwd_ret_h"].abs().quantile(label_quantile))
    return max(barrier, 1e-6)



def label_targets(df: pd.DataFrame, return_barrier: float) -> pd.DataFrame:
    out = df.copy()
    out["target_class"] = FLAT_CLASS
    out.loc[out["fwd_ret_h"] > return_barrier, "target_class"] = BUY_CLASS
    out.loc[out["fwd_ret_h"] < -return_barrier, "target_class"] = SELL_CLASS
    return out



def make_classifier(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )



def probability_columns(model: RandomForestClassifier) -> Dict[int, int]:
    return {int(cls): idx for idx, cls in enumerate(model.classes_)}



def derive_decision_thresholds(
    model: RandomForestClassifier,
    train_df: pd.DataFrame,
    prob_quantile: float,
    margin_quantile: float,
) -> Tuple[float, float, pd.DataFrame]:
    X_train = train_df[FEATURE_COLS].astype(np.float32)
    proba = model.predict_proba(X_train)
    colmap = probability_columns(model)

    p_sell = proba[:, colmap[SELL_CLASS]]
    p_flat = proba[:, colmap[FLAT_CLASS]]
    p_buy = proba[:, colmap[BUY_CLASS]]

    best_direction_prob = np.maximum(p_buy, p_sell)
    direction_is_buy = p_buy >= p_sell
    opposite_prob = np.where(direction_is_buy, p_sell, p_buy)
    best_vs_next = best_direction_prob - np.maximum(p_flat, opposite_prob)

    candidates = best_direction_prob > p_flat
    if candidates.any():
        entry_prob_threshold = float(np.quantile(best_direction_prob[candidates], prob_quantile))
        min_prob_gap = float(np.quantile(best_vs_next[candidates], margin_quantile))
    else:
        entry_prob_threshold = 0.55
        min_prob_gap = 0.05

    entry_prob_threshold = float(np.clip(entry_prob_threshold, 0.34, 0.95))
    min_prob_gap = float(np.clip(min_prob_gap, 0.00, 0.50))

    diag = pd.DataFrame(
        {
            "p_sell": p_sell,
            "p_flat": p_flat,
            "p_buy": p_buy,
            "best_direction_prob": best_direction_prob,
            "best_vs_next": best_vs_next,
            "target_class": train_df["target_class"].to_numpy(),
        }
    )
    return entry_prob_threshold, min_prob_gap, diag



def classify_with_thresholds(
    model: RandomForestClassifier,
    df: pd.DataFrame,
    entry_prob_threshold: float,
    min_prob_gap: float,
) -> pd.DataFrame:
    X = df[FEATURE_COLS].astype(np.float32)
    proba = model.predict_proba(X)
    colmap = probability_columns(model)

    p_sell = proba[:, colmap[SELL_CLASS]]
    p_flat = proba[:, colmap[FLAT_CLASS]]
    p_buy = proba[:, colmap[BUY_CLASS]]

    best_direction_prob = np.maximum(p_buy, p_sell)
    direction = np.where(p_buy >= p_sell, BUY_CLASS, SELL_CLASS)
    second_best = np.maximum(p_flat, np.where(direction == BUY_CLASS, p_sell, p_buy))
    prob_gap = best_direction_prob - second_best

    pred = np.full(len(df), FLAT_CLASS, dtype=np.int32)
    take_trade = (best_direction_prob >= entry_prob_threshold) & (prob_gap >= min_prob_gap) & (best_direction_prob > p_flat)
    pred[take_trade] = direction[take_trade]

    out = df.copy()
    out["p_sell"] = p_sell
    out["p_flat"] = p_flat
    out["p_buy"] = p_buy
    out["best_direction_prob"] = best_direction_prob
    out["prob_gap"] = prob_gap
    out["pred_class"] = pred
    out["trade_taken"] = out["pred_class"] != FLAT_CLASS
    out["direction_correct"] = (
        ((out["pred_class"] == BUY_CLASS) & (out["target_class"] == BUY_CLASS))
        | ((out["pred_class"] == SELL_CLASS) & (out["target_class"] == SELL_CLASS))
    )
    return out



def summarize_predictions(pred_df: pd.DataFrame) -> Dict[str, float]:
    y_true = pred_df["target_class"].to_numpy()
    y_pred = pred_df["pred_class"].to_numpy()
    trade_mask = pred_df["trade_taken"].to_numpy()

    accepted = pred_df.loc[trade_mask].copy()
    accepted_rate = float(trade_mask.mean())
    if len(accepted) > 0:
        directional_precision = float(accepted["direction_correct"].mean())
        mean_fwd_ret = float(
            np.where(accepted["pred_class"] == BUY_CLASS, accepted["fwd_ret_h"], -accepted["fwd_ret_h"]).mean()
        )
    else:
        directional_precision = 0.0
        mean_fwd_ret = 0.0

    # Ternary accuracy includes FLAT decisions too.
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist()
    return {
        "rows": int(len(pred_df)),
        "accepted_trades": int(trade_mask.sum()),
        "accepted_rate": accepted_rate,
        "directional_precision_on_trades": directional_precision,
        "mean_signed_fwd_return_on_trades": mean_fwd_ret,
        "ternary_accuracy": acc,
        "balanced_accuracy": bal_acc,
        "confusion_matrix_sell_flat_buy": cm,
    }



def walk_forward_report(
    train_df_raw: pd.DataFrame,
    label_quantile: float,
    prob_quantile: float,
    margin_quantile: float,
    n_splits: int = 5,
) -> Dict[str, float]:
    X = train_df_raw[FEATURE_COLS].astype(np.float32)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accepted_rates: List[float] = []
    precisions: List[float] = []
    mean_trade_returns: List[float] = []
    ternary_accs: List[float] = []
    balanced_accs: List[float] = []
    entry_thresholds: List[float] = []
    margin_thresholds: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        fold_train = train_df_raw.iloc[train_idx].copy()
        fold_valid = train_df_raw.iloc[valid_idx].copy()

        return_barrier = compute_return_barrier(fold_train, label_quantile)
        fold_train = label_targets(fold_train, return_barrier)
        fold_valid = label_targets(fold_valid, return_barrier)

        model = make_classifier(random_state=42)
        model.fit(fold_train[FEATURE_COLS].astype(np.float32), fold_train["target_class"].astype(np.int32))

        entry_prob_threshold, min_prob_gap, _ = derive_decision_thresholds(
            model,
            fold_train,
            prob_quantile=prob_quantile,
            margin_quantile=margin_quantile,
        )
        pred_valid = classify_with_thresholds(model, fold_valid, entry_prob_threshold, min_prob_gap)
        summary = summarize_predictions(pred_valid)

        accepted_rates.append(summary["accepted_rate"])
        precisions.append(summary["directional_precision_on_trades"])
        mean_trade_returns.append(summary["mean_signed_fwd_return_on_trades"])
        ternary_accs.append(summary["ternary_accuracy"])
        balanced_accs.append(summary["balanced_accuracy"])
        entry_thresholds.append(entry_prob_threshold)
        margin_thresholds.append(min_prob_gap)

        print(
            f"Fold {fold}: barrier={return_barrier:.6f} entry_prob={entry_prob_threshold:.4f} gap={min_prob_gap:.4f} "
            f"accepted_rate={summary['accepted_rate']:.3f} precision={summary['directional_precision_on_trades']:.3f} "
            f"mean_trade_ret={summary['mean_signed_fwd_return_on_trades']:.6f} bal_acc={summary['balanced_accuracy']:.3f}"
        )

    return {
        "accepted_rate_mean": float(np.mean(accepted_rates)),
        "directional_precision_mean": float(np.mean(precisions)),
        "mean_signed_fwd_return_on_trades_mean": float(np.mean(mean_trade_returns)),
        "ternary_accuracy_mean": float(np.mean(ternary_accs)),
        "balanced_accuracy_mean": float(np.mean(balanced_accs)),
        "entry_prob_threshold_mean": float(np.mean(entry_thresholds)),
        "prob_gap_threshold_mean": float(np.mean(margin_thresholds)),
    }



def export_to_onnx(model: RandomForestClassifier, output_path: Path) -> None:
    initial_type = [("float_input", FloatTensorType([1, len(FEATURE_COLS)]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=15,
        options={id(model): {"zipmap": False}},
    )
    output_path.write_bytes(onx.SerializeToString())



def save_metadata(
    output_dir: Path,
    symbol: str,
    timeframe: str,
    horizon_bars: int,
    train_ratio: float,
    label_quantile: float,
    return_barrier: float,
    entry_prob_threshold: float,
    min_prob_gap: float,
    walk_forward: Dict[str, float],
    train_summary: Dict[str, float],
    test_summary: Dict[str, float],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> None:
    meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "features": FEATURE_COLS,
        "model_type": "RandomForestClassifier",
        "class_order": CLASS_ORDER,
        "horizon_bars": horizon_bars,
        "train_ratio": train_ratio,
        "label_quantile": label_quantile,
        "return_barrier": return_barrier,
        "entry_prob_threshold": entry_prob_threshold,
        "min_prob_gap": min_prob_gap,
        "train_window_utc": {"start": train_start, "end": train_end},
        "test_window_utc": {"start": test_start, "end": test_end},
        "training_notes": {
            "target": f"3 classes on future {horizon_bars} bars: SELL / FLAT / BUY",
            "signal": "trade only if class probability >= entry_prob_threshold and probability gap >= min_prob_gap",
            "recommended_mt5_max_bars_in_trade": horizon_bars,
        },
        "walk_forward_on_train": walk_forward,
        "train_summary": train_summary,
        "test_summary": test_summary,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")



def save_readme_runfile(
    output_dir: Path,
    symbol: str,
    timeframe: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon_bars: int,
    entry_prob_threshold: float,
    min_prob_gap: float,
) -> None:
    text = f"""MODEL: RandomForestClassifier (3 classes)
SYMBOL: {symbol}
TIMEFRAME: {timeframe}
HORIZON TARGET (bars): {horizon_bars}

TRAIN UTC:
  start: {train_start}
  end  : {train_end}

TEST UTC:
  start: {test_start}
  end  : {test_end}

RECOMMENDED INPUTS FOR EA:
  InpEntryProbThreshold = {entry_prob_threshold:.6f}
  InpMinProbGap        = {min_prob_gap:.6f}
  InpMaxBarsInTrade    = {horizon_bars}

STEPS TO RUN IN MT5:
1. Copy ml_strategy_classifier.onnx file next to the EA .mq5 file.
2. Recompile the EA in MetaEditor.
3. Run the Strategy Tester ONLY on the TEST UTC window above.
"""
    (output_dir / "run_in_mt5.txt").write_text(text, encoding="utf-8")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an Random Forest classifier for MT5 and export it to ONNX.")
    p.add_argument("--symbol", default="XAGUSD", help="Symbol used for training")
    p.add_argument("--timeframe", default="M15", help="M1/M5/M15/M30/H1/H4/D1")
    p.add_argument("--bars", type=int, default=20000, help="Number of bars to read from MT5")
    p.add_argument("--csv", type=str, default="", help="Alternatively, read data from CSV")
    p.add_argument("--output-dir", default="output_v2", help="Output directory")
    p.add_argument("--horizon-bars", type=int, default=8, help="Target horizon in bars (recommended between 4 and 12)")
    p.add_argument("--train-ratio", type=float, default=0.70, help="Chronological percentage used for training")
    p.add_argument("--label-quantile", type=float, default=0.60, help="Quantile of abs(fwd_ret_h) above which we label BUY/SELL")
    p.add_argument("--prob-quantile", type=float, default=0.70, help="Quantile of training probabilities used to derive the entry threshold")
    p.add_argument("--gap-quantile", type=float, default=0.60, help="Quantile of the top1-top2 gap used to filter decisions")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        raw = load_rates_from_csv(Path(args.csv))
    else:
        raw = fetch_rates_from_mt5(args.symbol, args.timeframe, args.bars)

    raw = raw.sort_values("time").reset_index(drop=True)
    raw.to_csv(output_dir / "rates_snapshot.csv", index=False)

    feat_df = build_features(raw, horizon_bars=args.horizon_bars)
    feat_df.to_csv(output_dir / "features_snapshot.csv", index=False)

    train_df_raw, test_df_raw = split_train_test(feat_df, args.train_ratio)
    train_df_raw.to_csv(output_dir / "train_features_snapshot.csv", index=False)
    test_df_raw.to_csv(output_dir / "test_features_snapshot.csv", index=False)

    print(f"Total set with features: {len(feat_df)} rows")
    print(f"Train: {len(train_df_raw)} rows | Test: {len(test_df_raw)} rows")
    print(f"Train window: {train_df_raw['time'].iloc[0]} -> {train_df_raw['time'].iloc[-1]}")
    print(f"Test window : {test_df_raw['time'].iloc[0]} -> {test_df_raw['time'].iloc[-1]}")

    walk_forward = walk_forward_report(
        train_df_raw,
        label_quantile=args.label_quantile,
        prob_quantile=args.prob_quantile,
        margin_quantile=args.gap_quantile,
    )
    print("\nWalk-forward summary on train:")
    print(json.dumps(walk_forward, indent=2))

    return_barrier = compute_return_barrier(train_df_raw, args.label_quantile)
    train_df = label_targets(train_df_raw, return_barrier)
    test_df = label_targets(test_df_raw, return_barrier)

    model = make_classifier(random_state=42)
    model.fit(train_df[FEATURE_COLS].astype(np.float32), train_df["target_class"].astype(np.int32))

    entry_prob_threshold, min_prob_gap, train_diag = derive_decision_thresholds(
        model,
        train_df,
        prob_quantile=args.prob_quantile,
        margin_quantile=args.gap_quantile,
    )

    train_pred = classify_with_thresholds(model, train_df, entry_prob_threshold, min_prob_gap)
    test_pred = classify_with_thresholds(model, test_df, entry_prob_threshold, min_prob_gap)
    train_summary = summarize_predictions(train_pred)
    test_summary = summarize_predictions(test_pred)

    print(f"\nLabeling barrier for abs(fwd_ret_h): {return_barrier:.8f}")
    print(f"Entry probability threshold derived from train predictions: {entry_prob_threshold:.6f}")
    print(f"Probability gap threshold derived from train predictions: {min_prob_gap:.6f}")
    print("\nTrain summary:")
    print(json.dumps(train_summary, indent=2))
    print("\nTest summary:")
    print(json.dumps(test_summary, indent=2))

    onnx_path = output_dir / "ml_strategy_classifier.onnx"
    export_to_onnx(model, onnx_path)
    train_diag.to_csv(output_dir / "train_prediction_diagnostics.csv", index=False)
    train_pred.to_csv(output_dir / "train_predictions.csv", index=False)
    test_pred.to_csv(output_dir / "test_predictions.csv", index=False)

    save_metadata(
        output_dir=output_dir,
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizon_bars=args.horizon_bars,
        train_ratio=args.train_ratio,
        label_quantile=args.label_quantile,
        return_barrier=return_barrier,
        entry_prob_threshold=entry_prob_threshold,
        min_prob_gap=min_prob_gap,
        walk_forward=walk_forward,
        train_summary=train_summary,
        test_summary=test_summary,
        train_start=str(train_df_raw["time"].iloc[0]),
        train_end=str(train_df_raw["time"].iloc[-1]),
        test_start=str(test_df_raw["time"].iloc[0]),
        test_end=str(test_df_raw["time"].iloc[-1]),
    )
    save_readme_runfile(
        output_dir=output_dir,
        symbol=args.symbol,
        timeframe=args.timeframe,
        train_start=str(train_df_raw["time"].iloc[0]),
        train_end=str(train_df_raw["time"].iloc[-1]),
        test_start=str(test_df_raw["time"].iloc[0]),
        test_end=str(test_df_raw["time"].iloc[-1]),
        horizon_bars=args.horizon_bars,
        entry_prob_threshold=entry_prob_threshold,
        min_prob_gap=min_prob_gap,
    )

    print(f"\nONNX model saved in: {onnx_path}")
    print(f"Use in EA InpEntryProbThreshold = {entry_prob_threshold:.6f}")
    print(f"Use in EA InpMinProbGap        = {min_prob_gap:.6f}")
    print(f"Use in EA InpMaxBarsInTrade    = {args.horizon_bars}")
    print("Also read the run_in_mt5.txt file from the output for the exact test window.")


if __name__ == "__main__":
    main()
