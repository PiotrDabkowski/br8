#!/bin/csh
cd ~/br8
#$ -cwd
#$ -e error.err
THEANO_FLAGS=device=gpu,floatX=float32 python syntrain.py