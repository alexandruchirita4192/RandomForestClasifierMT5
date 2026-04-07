# MT5 ML Classifier v2

Pachetul contine o varianta refacuta a exemplului initial:

- target pe **mai multe bare** (`--horizon-bars`, recomandat 4..12)
- **clasificare** in 3 clase: `SELL`, `FLAT`, `BUY`
- **train/test separat nativ** in scriptul Python
- praguri pentru EA **derivate din predictiile modelului pe train**
- export ONNX pentru rulare in MT5 Strategy Tester

## Fisiere

- `train_mt5_ml_classifier.py`
- `MT5_ML_Classifier_ONNX_Strategy.mq5`

## Instalare pachete Python

```powershell
pip install MetaTrader5 pandas numpy scikit-learn skl2onnx onnx
```

## Rulare recomandata

Exemplu pentru XAGUSD, M15, 20000 bare, orizont 8 bare:

```powershell
python train_mt5_ml_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_v2_xagusd_m15_h8
```

Dupa rulare, in directorul de output vei avea:

- `ml_strategy_classifier.onnx`
- `model_metadata.json`
- `run_in_mt5.txt`
- `train_predictions.csv`
- `test_predictions.csv`

## Cum rulezi in MT5

1. Copiezi `ml_strategy_classifier.onnx` langa `MT5_ML_Classifier_ONNX_Strategy.mq5`
2. Recompilezi EA-ul in MetaEditor
3. Deschizi `run_in_mt5.txt`
4. In Strategy Tester setezi exact fereastra `TEST UTC`
5. In inputurile EA folosesti valorile recomandate pentru:
   - `InpEntryProbThreshold`
   - `InpMinProbGap`
   - `InpMaxBarsInTrade`

## Ce sa compari

Merita sa compari cel putin 3 variante:

```powershell
python train_mt5_ml_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 4  --train-ratio 0.70 --output-dir output_v2_h4
python train_mt5_ml_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8  --train-ratio 0.70 --output-dir output_v2_h8
python train_mt5_ml_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 12 --train-ratio 0.70 --output-dir output_v2_h12
```

Apoi backtest in MT5 doar pe ferestrele de test corespunzatoare.
