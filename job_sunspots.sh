#!/bin/sh
module load cesga/2020 python/3.9.9
export PYTHONNOUSERSITE=1
source $STORE/pballs/bin/activate
lscpu
python sunspots/iso_sunspots_cycles.py $block
git add .
git commit -m "Sunspots simulations block $block"
git push

# For submitting all blocks (0-8):
# for b in {0..8}
# do
#   sbatch --time=02:10:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="%j-sunspots-block$b.out" job_sunspots.sh
# done

# Example single block submission:
# sbatch --time=02:10:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=0 --output="%j-sunspots-block0.out" job_sunspots.sh
