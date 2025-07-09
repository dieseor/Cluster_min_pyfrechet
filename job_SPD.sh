#!/bin/sh
module load cesga/2020 python/3.9.9
export PYTHONNOUSERSITE=1
source $STORE/pballs/bin/activate
lscpu
python simulations_SPD/ALL_main_parallel.py $block
git add .
git commit -m "Simulations SPD block $block"
git push

# for b in {6..100}
# do
#   sbatch --time=03:40:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="%j-SPD-block$b.out" job_SPD.sh
# done

# sbatch --time=03:40:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=1 --output="%j-SPD-block1.out" job_SPD.sh 