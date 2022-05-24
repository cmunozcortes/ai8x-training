#!/bin/sh
python train.py --epochs 5 --optimizer Adam --lr 0.00030 --batch-size 256 \
--deterministic --compress schedule-asl.yaml --model ai85simplenetasl \
--dataset asl_mnist --confusion --device MAX78000 --use-bias "$@"