#!/bin/bash
# Replicate the primary-split models on the two other registered-domain-disjoint split seeds (22, 33).
cd "$(dirname "$0")/.."
for S in 22 33; do
  python3 scripts/v2_train_xlin.py --seed $S --variants xlin_pw1_base xlin_pw2_base xlin_pw2_full xlin_pw2_aug
  python3 scripts/v2_train_charcnn.py --seed $S --pos-weight 2 --epochs 4 --budget-min 30
done
echo CHAIN_DONE
