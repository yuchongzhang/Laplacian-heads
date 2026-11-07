# Language modeling

- `nanochat/gpt.py`: architecture definitions.
- `scripts/base_train.py`, `scripts/base_eval.py`: training, validation bits per byte (BPB), and CORE evaluation.
- `nanochat/dataset.py`, `scripts/tok_train.py`: ClimbMix download and BPE tokenizer training.
- `nanochat/measurements/`: context preparation, post-attention residual features, spectra, and next-token ANOVA.
- `scripts/plot_measurements.py`, `scripts/plot_performance.py`: figures and CSV tables from saved measurements.
- `LICENSE`: MIT license for the [nanochat](https://github.com/karpathy/nanochat) code.

## Setup and data

Run from the repository root:

```bash
bash setup/bootstrap.sh
source .venv/bin/activate
cd nanochat
export NANOCHAT_BASE_DIR=/path/to/nanochat-artifacts
python -m nanochat.dataset -n 380 -w 8
python -m scripts.tok_train --vocab-size=32768
```

Checkpoint evaluation requires its matching `tokenizer/tokenizer.pkl` and `tokenizer/token_bytes.pt` in `$NANOCHAT_BASE_DIR`.

## Training and evaluation

```bash
DEPTH=12
HEADS=3
SEED=1
case $DEPTH in
  12) BATCH=32; EVAL=250; SAVE=-1 ;;
  20) BATCH=16; EVAL=250; SAVE=-1 ;;
  30) BATCH=16; EVAL=800; SAVE=2000 ;;
esac
TAG=d${DEPTH}-lap${HEADS}-vanilla-s${SEED}
torchrun --standalone --nproc_per_node=4 -m scripts.base_train \
  --depth=$DEPTH --laplacian-heads=$HEADS --seed=$SEED --model-tag=$TAG \
  --fp8 --window-pattern=SSSL \
  --target-param-data-ratio=12 --device-batch-size=$BATCH \
  --eval-every=$EVAL --eval-tokens=10485760 --save-every=$SAVE

torchrun --standalone --nproc_per_node=4 -m scripts.base_eval \
  --model-tag=$TAG --eval=core,bpb --device-batch-size=16 \
  --split-tokens=20971520 --max-per-task=-1 --seed=1
```

```bash
python -m scripts.plot_performance \
  "$NANOCHAT_BASE_DIR"/base_eval/*/base_model_*.json --output-dir outputs/performance
```

## Representation measurements

Features are the RMS-normalized residual stream after attention in the second-to-last block (`N-2`), captured at its MLP input.

Prepare the measurement inputs from the validation shard:

```bash
SHARD="$NANOCHAT_BASE_DIR/base_data_climbmix/shard_06542.parquet"
python -m scripts.prepare_measurement_data \
  --input "$SHARD" --output outputs/contexts.npy
```

```bash
CHECKPOINT="$NANOCHAT_BASE_DIR/base_checkpoints/$TAG"
python -m scripts.measure_representations \
  --checkpoint-dir "$CHECKPOINT" --data outputs/contexts.npy \
  --output outputs/measurements/${TAG}.json
python -m scripts.plot_measurements outputs/measurements/*.json \
  --output-dir outputs/figures
```

`--step=STEP` selects a saved checkpoint; `--batch-size` controls inference memory.
