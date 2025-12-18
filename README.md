# ESE5460-Project-LinearCrypto

## Train
```bash
python scripts/training.py \
  --config <CONFIG_NAME> \
  --save_checkpoints
```

## Evaluate
```bash
python scripts/evaluation.py \
  --config <CONFIG_NAME> \
  --ckpt_path <CKPT_PATH> \
```

## Simulate trade
```bash
python scripts/simulate_trade.py \
  --config <CONFIG_NAME> \
  --ckpt_path <CKPT_PATH> \
  --split <train|val|test> \
  --trade_mode <smart|smart_w_short|vanilla|no_strategy> \
```

## Acknowledgements

This project is partially based on the open-source project
[**CryptoMamba**](https://github.com/MShahabSepehri/CryptoMamba).

We adapt and extend the original implementation for the ESE5460 course project.

