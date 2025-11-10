#!/bin/sh
python train.py --model ai85hvsnet --dataset hammer_vs_screwdriver --data data/hammer_vs_screwdriver --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-hammerscrewdriver-qat8-q.pth.tar -8 --device MAX78000 "$@"
