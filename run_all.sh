#!/bin/bash
# Run all combinations: 2 models × 7 optimizers × 1 seed = 14 runs

#MODELS="mlp resnet"
MODELS="mlp"
OPTS="sgd adamw hyperadam muon hypermuon_l1 hypermuon_l2 hypermuon_l3"
SEEDS="0"

for model in $MODELS; do
  for opt in $OPTS; do
    for seed in $SEEDS; do
      echo "=== Running: model=$model | optimizer=$opt | seed=$seed ==="
      python train.py --model "$model" --optimizer "$opt" --seed "$seed"
    done
  done
done

echo ""
echo "=== All runs complete. Generating figures... ==="
python plot.py
