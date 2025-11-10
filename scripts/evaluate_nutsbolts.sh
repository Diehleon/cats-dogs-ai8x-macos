#!/bin/sh
python -W "ignore:pkg_resources is deprecated as an API" train.py --model ai85nutsboltsnet --dataset nuts_vs_bolts --data ./data --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-nutsbolts-qat8-q.pth.tar -8 --device MAX78000 "$@"
