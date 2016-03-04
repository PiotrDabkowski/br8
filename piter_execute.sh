#!/usr/bin/env bash
cd ~/br8
THEANO_FLAGS=device=gpu,floatX=float32 python memtrain.py