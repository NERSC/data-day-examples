#!/bin/bash -l

#SBATCH -N 2
#SBATCH -t 30

module load spark/2.0.0
start-all.sh

spark-submit spark-astro-ml.py
