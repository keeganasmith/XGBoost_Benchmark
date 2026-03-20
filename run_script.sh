#!/bin/bash

export DATA=/scratch/user/u.ks124812/TLC-Trip-Record-Data-Scripts/yellow_all_preprocessed.parquet
export VAL_SPLIT=.1
export CORES=96
export TARGET=total_amount
export OMP_NUM_THREADS=96

python3 main.py --data $DATA --val_split $VAL_SPLIT --cores $CORES --target $TARGET


