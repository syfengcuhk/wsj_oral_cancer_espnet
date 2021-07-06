#!/bin/bash
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=04:00:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=4
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END
##SBATCH --nodelist=ewi1

#srun bash run.sh --stage 4 --stop-stage 5 --ngpu 2 --resume "exp/train_si284_pytorch_train_no_preprocess/results/snapshot.ep.26" 


# Apply ASR model retraining
ID=5 # 1 ~ 5, using which parition of OC training set to retrain the WSJ ASR model.
#srun bash run.sh --stage 9 --stop-stage 10  --ngpu 1 --retrain-partition $ID --retrain-config "conf/tuning/retrain_pytorch_transformer_ep20_lr0.5_wu250.yaml"
srun bash run.sh --stage 9 --stop-stage 10  --ngpu 1 --retrain-partition $ID --retrain-config "conf/tuning/retrain_pytorch_transformer_ep20_lr0.5_wu10.yaml"

#[low priority, only when e2e always worse than hybrid] Apply specaugment

#srun bash run.sh --stage 4 --stop-stage 5 --ngpu 2  --preprocess-config "conf/specaug_F8.yaml"
