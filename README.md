# MT5 ML Classifier v2

The package contains a refactored version of the initial example:

- target on **multiple bars** (`--horizon-bars`, recommended 4..12)
- **classification** into 3 classes: `SELL`, `FLAT`, `BUY`
- **native train/test split** in the Python script
- thresholds for EA **derived from model predictions on train**
- ONNX export for running in MT5 Strategy Tester

## Files

- `train_mt5_ml_classifier.py`
- `MT5_ML_Classifier_ONNX_Strategy.mq5`

## Python packages installation

```powershell
pip install MetaTrader5 pandas numpy scikit-learn skl2onnx onnx
```

## Recommended run

Example for XAGUSD, M15, 20000 bars, horizon 8 bars:

```powershell
python train_mt5_ml_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_v2_xagusd_m15_h8
```

After running, in the output directory you will have:

- `ml_strategy_classifier.onnx`
- `model_metadata.json`
- `run_in_mt5.txt`
- `train_predictions.csv`
- `test_predictions.csv`

## How to run in MT5

1. Copy `ml_strategy_classifier.onnx` next to `MT5_ML_Classifier_ONNX_Strategy.mq5`
2. Recompile the EA in MetaEditor
3. Open `run_in_mt5.txt`
4. In Strategy Tester set exactly the `TEST UTC` window
5. In the EA inputs use the recommended values for:
   - `InpEntryProbThreshold`
   - `InpMinProbGap`
   - `InpMaxBarsInTrade`

## What to compare

It is worth comparing at least 3 variants:

```powershell
python train_mt5_ml_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 4  --train-ratio 0.70 --output-dir output_v2_h4
python train_mt5_ml_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8  --train-ratio 0.70 --output-dir output_v2_h8
python train_mt5_ml_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 12 --train-ratio 0.70 --output-dir output_v2_h12
```

Then backtest in MT5 only on the corresponding test windows.
