# ESE5460-Project-LinearCrypto

## Train
```bash
python scripts/training.py \
  --config <CONFIG_NAME> \
  --accelerator <gpu|cpu> \
  --devices <NUM_DEVICES> \
  --batch_size <BATCH_SIZE> \
  --logger_type <tb|wandb> \
  --save_checkpoints
```

## Evaluate
```bash
python scripts/evaluation.py \
  --config <CONFIG_NAME> \
  --ckpt_path <CKPT_PATH> \
  --accelerator <gpu|cpu> \
  --devices <NUM_DEVICES> \
  --batch_size <BATCH_SIZE>
```

## Simulate trade
```bash
python scripts/simulate_trade.py \
  --config <CONFIG_NAME> \
  --ckpt_path <CKPT_PATH> \
  --split <train|val|test> \
  --trade_mode <smart|smart_w_short|vanilla|no_strategy> \
  --balance <INITIAL_BALANCE> \
  --risk <RISK_PERCENT>
```
