#!/bin/bash

source activate pystan
#source activate r
export TMPDIR=$HOME/tmp
#export R_LIBS_USER=/home/teamshare/testing-centre/anaconda/lib/R/library:$HOME/R/x86_64-pc-linux-gnu-library/3.1/

ncpu=20
pname=big,ssd,main
func=precision
kpi=("normal_shifted")
#outdir=parameter_estimation

rope_width=(0.06 0.08 0.1 0.12 0.14)
hdi_mass=(0.8 0.85 0.9 0.95)
distribution=("cauchy" "normal")
scale=(1 2 5 10)
eid=("5609412b5c6ee5f32c12f85d" "56b8a44026f9d8072e54a73d")
#method=("bayes_factor" "bayes_precision") # ("group_sequential")
method=("group_sequential") #("precision" "group_sequential")

if [ ! -d "$func" ]; then
	mkdir $func
fi

#for rw in "${rope_width[@]}"; do
#	for hm in "${hdi_mass[@]}"; do
# 		srun --job-name=stan --cpus-per-task=$ncpu --mem-per-cpu=10000 --partition=$pname -o log/job_${func}_${kpi}.log -D $HOME/src/early-stopping \
# 			python simulate.py \
# 			-c $ncpu \
# 			-f $func \
# 			-k $kpi \
# 			-m $HOME/src/early-stopping/normal_kpi_template.stan \
# 			-w $rw \
# 			-d $hm &
# 	done
# done

for k in "${kpi[@]}"; do
	for m in "${method[@]}"; do
		srun --job-name=$m --cpus-per-task=$ncpu --mem-per-cpu=10000 --partition=$pname -o log/job_${k}_${m}.log -D $HOME/src/early-stopping \
			python simulate.py \
			-c $ncpu \
			-f $m \
			-k $k \
			-m $HOME/src/early-stopping/normal_kpi_template.stan \
			--distribution cauchy \
			--scale 1 &
	done
done

# for e in "${eid[@]}"; do
# 	for m in "${method[@]}"; do
# 		srun --job-name=bunchbox --cpus-per-task=$ncpu --mem-per-cpu=10000 --partition=$pname -o log/job_${e}_${m}.log -D $HOME/src/early-stopping \
# 			python bunchbox_data.py \
# 			-c $ncpu \
# 			-e $e \
# 			-m $m &
# 	done
# done

# srun --job-name=$func --cpus-per-task=$ncpu --mem-per-cpu=10000 --partition=$pname -o log/job_${func}.log -D $HOME/src/early-stopping \
# 	python simulate.py \
# 	-c $ncpu \
# 	-f $func \
# 	-k $kpi \
# 	-m $HOME/src/early-stopping/normal_kpi.stan \
#  	--distribution cauchy \
#  	--scale 1 &

# srun --job-name=stan --cpus-per-task=$ncpu --mem-per-cpu=10000 --partition=$pname -o log/job_${func}.log -D $HOME/src/early-stopping \
# 	Rscript --default-packages=methods,utils simulate.R \
# 	$ncpu \
# 	$func \
# 	$kpi \
# 	$HOME/src/early-stopping/normal_kpi.stan \
# 	0.1 \
# 	0.95 &