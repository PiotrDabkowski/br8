#!/bin/csh
cd ~/br8
#$ -e error.err
THEANO_FLAGS=device=gpu,floatX=float32 python syntrain.py