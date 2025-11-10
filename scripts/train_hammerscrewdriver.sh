#!/bin/sh
python -W "ignore:pkg_resources is deprecated as an API" train.py --epochs 50 --optimizer Adam --lr 0.0005 --wd 0.001 --deterministic --compress policies/schedule-hammer_screwdriver.yaml --qat-policy policies/qat_policy_hammerscrewdriver.yaml --model ai85hvsnet --dataset hammer_vs_screwdriver --data ./data/hammer_vs_screwdriver --confusion --param-hist --embedding --device MAX78000 "$@"
