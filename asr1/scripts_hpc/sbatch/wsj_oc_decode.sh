#!/bin/bash
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=03:00:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=30
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=70G
##SBATCH --gres=gpu
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END
##SBATCH --nodelist=ewi1

set=test_eval92
#set=test_dev93
#srun bash run.sh --stage 6 --stop-stage 7 --nj-decode 45 --test-recog-set "$set"

##########Oral cancer ASR decoding
ID=5 # 2 3 4 5
test_or_train=train #test
srun bash run.sh --stage 8 --stop-stage 9 --test-oc-partition $ID --nj-decode 30 --oc-test-or-train $test_or_train

# Tune decoding parameters, lm-weight and ctc-weight
lw=0.5 # by default:1.0; 0.5 found worse than 1.0
decode_tag="_lw$lw"
extra_rec_config="--lm-weight $lw"
#srun bash run.sh --stage 8 --stop-stage 9 --test-oc-partition $ID --nj-decode 30 --extra-rec-config "$extra_rec_config" --decode-tag $decode_tag


#####NOT IMPORTANT
# What if use letter RNNLM instead of word RNNLM?
#srun bash run.sh --stage 6 --stop-stage 7 --nj-decode 45 --test-recog-set "$set" --use-wordlm false



