#!/bin/sh
module load cesga/2020 python/3.9.9
export PYTHONNOUSERSITE=1
source $STORE/pballs/bin/activate
lscpu
python simulations_sphere/prueba.py
git add .
git commit -m "pruebas"
git push
