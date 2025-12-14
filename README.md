# ESE5460-Project-LinearCrypto

## Train
python scripts/training.py \
  --config <CONFIG_NAME> \
  --save_checkpoints

## Evaluate
python scripts/evaluation.py \
  --config <CONFIG_NAME> \
  --ckpt_path <CKPT_PATH> \

## Simulate trade
python scripts/simulate_trade.py \
  --config <CONFIG_NAME> \
  --ckpt_path <CKPT_PATH> \
  --split <train|val|test> \
  --trade_mode <smart|smart_w_short|vanilla|no_strategy> \

