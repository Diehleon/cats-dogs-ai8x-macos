#!/bin/sh
# Training script for Nuts vs Bolts

python -W "ignore:pkg_resources is deprecated as an API" train.py \
  --model ai85nutsboltsnet \
  --dataset nuts_vs_bolts \
  --data ./data\
  --epochs 100 \
  --optimizer Adam \
  --lr 0.001 \
  --wd 0.0005 \
  --deterministic \
  --compress policies/schedule-nuts_vs_bolts.yaml \
  --qat-policy policies/qat_policy_nuts_vs_bolts.yaml \
  --confusion \
  --param-hist \
  --embedding \
  --device MAX78000 \
  "$@"