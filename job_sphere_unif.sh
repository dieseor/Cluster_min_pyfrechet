#!/bin/sh
module load cesga/2020 python/3.9.9
export PYTHONNOUSERSITE=1
source $STORE/pballs/bin/activate
lscpu
python simulations_sphere/sphere_parallel_unif.py $block
git add .
git commit -m "Simulations Sphere block $block"
git push

# for b in {2..40}
# do
#   sbatch --time=02:10:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="%j-sphere-block$b.out" job_sphere.sh
# done

# sbatch --time=02:40:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=1 --output="%j-sphere-block1.out" job_sphere.sh
