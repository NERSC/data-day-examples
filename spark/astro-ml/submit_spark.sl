#!/bin/bash -l

#SBATCH -N 2
#SBATCH -t 30

module load spark
start-all.sh

spark-submit spark-astro-ml
