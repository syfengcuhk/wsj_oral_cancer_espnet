#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2021 Delft University of Technology (Siyuan Feng)
# This script is adapted from egs/wsj/asr1/run.sh
# This script implements experiments on oral cancer speech ASR 
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1        # start from 0 if you need to start from data preparation
stop_stage=2
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1
nj=4
nj_decode=4
if_gpu_decoding=false
ngpu_decode=0
batchsize_decoding=0
extra_rec_config="" # e.g. "--ctc-weight 0.3 "
decode_tag=""
# feature configuration
do_delta=false

# sample filtering
min_io_delta=4  # samples with `len(input) - len(output) * min_io_ratio < min_io_delta` will be removed.

# config files
preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
skip_lm_training=false  # for only using end-to-end ASR model without LM
use_wordlm=true         # false means to train/use a character LM
lm_vocabsize=65000      # effective only for word LMs
lm_resume=              # specify a snapshot file to resume LM training
lmtag=                  # tag for managing LMs

# decoding parameter
recog_model=model.acc.best   # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=10                 # the number of ASR models to be averaged
use_valbest_average=false    # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
constraint="" # if e.g. set as "1[0-5]", then only concern snapshot.ep.10 ~ snapshot.ep.15
# data
#wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=data/wsj/ #/export/corpora5/LDC/LDC94S13B

# exp tag
tag="" # tag for managing experiments.
kaldi_wsj_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/wsj_s5
test_recog_set="test_dev93 test_eval92"
test_oc_partition=1
oc_test_or_train=test

# retrain related
retrain_partition=1
resume_retrain=""
retrain_config=conf/retrain.yaml
. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
#set -u
#set -o pipefail

train_set=train_si284
train_dev=test_dev93
train_test=test_eval92
recog_set="test_dev93 test_eval92"

if [ ${stage} -le 0 ] && [ ${stop_stage} -gt 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -gt 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
#    for x in train_si284 test_dev93 test_eval92; do
##        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
##            data/${x} exp/make_fbank/${x} ${fbankdir}
#        # directly copy features from kaldi dir
#        utils/copy_data_dir.sh $kaldi_wsj_root/data-fbank-pitch/$x data/$x
#        utils/fix_data_dir.sh data/${x}
#    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj $nj --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -gt 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done

    ### Filter out short samples which lead to `loss_ctc=inf` during training
    ###  with the specified configuration.
    # Samples satisfying `len(input) - len(output) * min_io_ratio < min_io_delta` will be pruned.
    local/filtering_samples.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --data-json ${feat_tr_dir}/data.json \
        --mode-subsample "asr" \
        ${min_io_delta:+--min-io-delta $min_io_delta} \
        --output-json-path ${feat_tr_dir}/data.json
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this by setting skip_lm_training=true
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi

lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -gt 3 ] && ! ${skip_lm_training}; then
    echo "stage 3: LM Preparation"


    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
#        mkdir -p ${lmdatadir}
#        cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train_trans.txt
#        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
#                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
#        cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
#        cut -f 2- -d" " data/${train_test}/text > ${lmdatadir}/test.txt
#        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
#        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
#        mkdir -p ${lmdatadir}
#        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
#            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
#        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
#            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
#            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
#        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
#            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
#        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_test}/text \
#                | cut -f 2- -d" " > ${lmdatadir}/test.txt
#        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi

    echo "LM Training"
    echo "Use word LM: ${use_wordlm}"
    echo "dict: $lmdict"
    echo "dir: $lmexpdir"
    echo "machine: $(hostname)"
    echo "number of GPUs: $ngpu"
    echo "training data: ${lmdatadir}/train.txt"
    echo "cross-validation data: ${lmdatadir}/valid.txt; test data: $lmdatadir/test.txt"
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -gt 4 ]; then
    echo "stage 4: Network Training"

    echo "dir: $expdir"
    echo "machine: $(hostname)"
    echo "number of GPUs: $ngpu"
    echo "training data: ${feat_tr_dir}"
    echo "cross-validation data: ${feat_dt_dir}"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
   [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
   [[ $(get_yaml.py ${train_config} model-module) = *maskctc* ]] || \
   [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
   [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
    average_opts=
    if ${use_valbest_average}; then
        recog_model=model.val${n_average}.avg.best
        average_opts="--log ${expdir}/results/log"
    else
        recog_model=model.last${n_average}.avg.best
    fi
fi
if [ ${stage} -le 5 ] && [ ${stop_stage} -gt 5 ]; then
    echo "stage 5: Get averaged model: $recog_model; average_opts: $average_opts"
#    nj=32
    average_checkpoints.py --backend ${backend} \
                           --snapshots ${expdir}/results/snapshot.ep.* \
                           --out ${expdir}/results/${recog_model} \
                           --num ${n_average} \
                           ${average_opts}
fi

nj=$nj_decode
if $if_gpu_decoding; then
  ngpudecode=${ngpu_decode}
  api=v2
else
  # Default:
  api=v1
  ngpudecode=0
  batchsize_decoding=0
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -gt 6 ]; then
    echo "Decoding: Use GPU to decode: ${if_gpu_decoding}; api: $api"
    echo "ngpu: ${ngpudecode}; batch size in decoding: ${batchsize_decoding}"
    echo "num jobs: $nj" 
    echo "ASR model: ${expdir}/results/${recog_model}"
    echo "LM model: ${lmexpdir}/$lang_model"
#    pids=() # initialize pids
    for rtask in ${test_recog_set}; do
#    (
        recog_opts=
        if ${skip_lm_training}; then
            if [ -z ${lmtag} ]; then
                lmtag="nolm"
            fi
        else
            if [ ${use_wordlm} = true ]; then
                recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
            else
                recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
            fi
        fi
        echo "Set: $rtask"
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json


        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpudecode} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --batchsize $batchsize_decoding \
            --api $api \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

 #   ) &
#    pids+=($!) # store background pids
    done
fi

oc_data_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/TUD_for_journal/data_fbank_pitch_for_kaldi_partition_5fold
if [ ${stage} -le 7 ] && [ ${stop_stage} -gt 7 ]; then
    echo "Prepare oral cancer speech test data  of all the 5 partitions"
    for partition in 1 2 3 4 5; do
      utils/copy_data_dir.sh  $oc_data_root/$partition/train data/oral_cancer/$partition/train
      utils/copy_data_dir.sh $oc_data_root/$partition/test data/oral_cancer/$partition/test
      # Reason: Utterance ID 10130025 has empty transcript so remove this
      sed -i '/10130025/d' data/oral_cancer/$partition/train/text
      utils/fix_data_dir.sh data/oral_cancer/$partition/train
      sed -i '/10130025/d' data/oral_cancer/$partition/test/text
      utils/fix_data_dir.sh data/oral_cancer/$partition/test
         
    done

    echo "Dump features"
    for partition in 1 2 3 4 5; do
      for set in train test; do
        feat_recog_dir=${dumpdir}/oral_cancer/$partition/$set/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data/oral_cancer/${partition}/$set/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/oral_cancer/$partition/$set \
            ${feat_recog_dir}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/oral_cancer/$partition/$set ${dict} > ${feat_recog_dir}/data.json
      done
    done

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -gt 8 ]; then
  echo "Decoding oral cancer speech"
  echo "Decoding: Use GPU to decode: ${if_gpu_decoding}; api: $api"
  echo "ngpu: ${ngpudecode}; batch size in decoding: ${batchsize_decoding}"
  echo "num jobs: $nj" 
  echo "ASR model: ${expdir}/results/${recog_model}"
  echo "LM model: ${lmexpdir}/$lang_model"
  for partition_id in ${test_oc_partition}; do
    recog_opts=
    if ${skip_lm_training}; then
        if [ -z ${lmtag} ]; then
            lmtag="nolm"
        fi
    else
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
    fi
    echo "Oral cancer partition ID: $partition_id"
    if [ "$oc_test_or_train" = "train" ]; then
      decode_flag=_train
    else
      decode_flag=""
    fi
    decode_dir=decode_oral_cancer${decode_flag}_${partition_id}_$(basename ${decode_config%.*})_${lmtag}${decode_tag}
    feat_recog_dir=${dumpdir}/oral_cancer/$partition_id/$oc_test_or_train/delta${do_delta}
    echo "Test or Train: $oc_test_or_train"
    echo "Decode dir: $decode_dir; $extra_rec_config"
    # split data
    splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
    ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpudecode} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --batchsize $batchsize_decoding \
            --api $api \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} $extra_rec_config  \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict} 
  done
fi
if [ ! "$retrain_config" = "conf/retrain.yaml" ]; then
expdir_retrain=${expdir}_retrain_by_oc_partition${retrain_partition}_$(basename ${retrain_config%.*})
else
expdir_retrain=${expdir}_retrain_by_oc_partition${retrain_partition}
fi
pretrain_dir=${expdir}
if [ ${stage} -le 9 ] && [ ${stop_stage} -gt 9 ]; then
    feat_tr_dir=${dumpdir}/oral_cancer/$retrain_partition/train/delta${do_delta}
    feat_dt_dir=${dumpdir}/oral_cancer/$retrain_partition/test/delta${do_delta}
    echo "stage 4: Network re-training with oral cancer training data"
    echo "dir: $expdir_retrain"
    echo "machine: $(hostname)"
    echo "number of GPUs: $ngpu"
    echo "training data: ${feat_tr_dir}"
    echo "cross-validation data: ${feat_dt_dir}"
    echo "pretrained model: $pretrain_dir/results/model.last10.avg.best"
    ${cuda_cmd} --gpu ${ngpu} ${expdir_retrain}/train.log \
        asr_train.py \
        --config ${retrain_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir_retrain}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir_retrain} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume_retrain} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --enc-init $pretrain_dir/results/model.last10.avg.best \
        --enc-init-mods 'encoder.' \
        --dec-init $pretrain_dir/results/model.last10.avg.best \
        --dec-init-mods 'decoder.,att'
fi

if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
   [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
   [[ $(get_yaml.py ${train_config} model-module) = *maskctc* ]] || \
   [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
   [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
    average_opts=
    if ${use_valbest_average}; then
        recog_model_retrain=model.val${n_average}.avg.best
        average_opts="--log ${expdir_retrain}/results/log"
    else
        recog_model_retrain=model.last${n_average}.avg.best
    fi
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -gt 10 ]; then
    echo "stage 10: Get averaged retrained model: $recog_model_retrain; average_opts: $average_opts"
#    nj=32
    if [ -n "$constraint" ]; then
      average_checkpoints.py --backend ${backend} \
                           --snapshots ${expdir_retrain}/results/snapshot.ep.${constraint} \
                           --out ${expdir_retrain}/results/${recog_model_retrain} \
                           --num ${n_average} \
                           ${average_opts}
    else
      average_checkpoints.py --backend ${backend} \
                           --snapshots ${expdir_retrain}/results/snapshot.ep.* \
                           --out ${expdir_retrain}/results/${recog_model_retrain} \
                           --num ${n_average} \
                           ${average_opts} 
    fi
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -gt 11 ]; then
  echo "Decoding oral cancer speech"
  echo "Decoding: Use GPU to decode: ${if_gpu_decoding}; api: $api"
  echo "ngpu: ${ngpudecode}; batch size in decoding: ${batchsize_decoding}"
  echo "num jobs: $nj" 
  echo "ASR model: ${expdir_retrain}/results/${recog_model_retrain}"
  echo "LM model: ${lmexpdir}/$lang_model"
  partition_id=$retrain_partition
  recog_opts=
  if ${skip_lm_training}; then
      if [ -z ${lmtag} ]; then
          lmtag="nolm"
      fi
  else
      if [ ${use_wordlm} = true ]; then
          recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
      else
          recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
      fi
  fi
  echo "Oral cancer partition ID: $partition_id"
  if [ "$oc_test_or_train" = "train" ]; then
    decode_flag=_train
  else
    decode_flag=""
  fi
  if [ "$recog_model" = "model.last5.avg.best" ]; then
    decode_dir=decode_oral_cancer${decode_flag}_${partition_id}_$(basename ${decode_config%.*})_${lmtag}${decode_tag}
  else
    decode_dir=decode_oral_cancer${decode_flag}_${partition_id}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}${decode_tag}
  fi
  feat_recog_dir=${dumpdir}/oral_cancer/$partition_id/$oc_test_or_train/delta${do_delta}
  echo "Test or Train: $oc_test_or_train"
  echo "Decode dir: $decode_dir; $extra_rec_config"
  # split data
  splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
  ${decode_cmd} JOB=1:${nj} ${expdir_retrain}/${decode_dir}/log/decode.JOB.log \
          asr_recog.py \
          --config ${decode_config} \
          --ngpu ${ngpudecode} \
          --backend ${backend} \
          --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
          --batchsize $batchsize_decoding \
          --api $api \
          --result-label ${expdir_retrain}/${decode_dir}/data.JOB.json \
          --model ${expdir_retrain}/results/${recog_model_retrain} $extra_rec_config  \
          ${recog_opts}

      score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir_retrain}/${decode_dir} ${dict} 
fi
echo "$0: Finished"
