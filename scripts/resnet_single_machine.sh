#!/usr/bin/env bash
set -euo pipefail
set -x
# 切到脚本所在目录，再切到项目根目录（按你的目录结构调整）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # 如果脚本在 scripts/ 下，项目根目录是上一级

# 确保 logs 目录存在
mkdir -p ./logs

export CUDA_VISIBLE_DEVICES=0
opts=("subscafsgd")
comp_dims=(3)
log_suffixes=("")
gene_methods=("cd")
epochs=(60)
batch_sizes=(128)
taus=(5)
lrs=0.1
warmups=100
cp_freqs=50
for ((i=0; i<${#opts[@]}; i++))
do
    for ((j=0; j<${#taus[@]}; j++))
    do
        opt=${opts[$i]}
        comp_dim=${comp_dims[0]}
        log_suffix=${log_suffixes[0]}
        gene_method=${gene_methods[0]}
        epoch=${epochs[0]}
        batch_size=${batch_sizes[0]}
        tau=${taus[$j]}
        torchrun --nproc-per-node 1 --master-port 25900 resnet_new_log.py \
            --comp_dim ${comp_dim} \
            --arch resnet110 \
            --optimizer ${opt} \
            --batch_size ${batch_size} \
            --tau ${tau} \
            --weight_decay 5e-4 \
            --epochs ${epoch} \
            --sanity_exp1 \
            --lr ${lrs} \
            --warmup ${warmups} \
            --update_cp_freq ${cp_freqs} \
            --use_alg4_momentum \
            --use_log \
            --evaluate \
            --print_freq 1 \
            --fix_P \
            --use_alg5 \
            --gene_method ${gene_method} \
            > "./logs/cifar100_alg5_testing_kernel_3_transform_weight_decay_5em4_fix_p_sanity_data_homo_test_new_log_${opt}${log_suffix}_${gene_method}_comp_${comp_dim}_batch_${batch_size}_epoch_60_tau_${tau}_no_constant_lr_01_warm_up_100_cp_freq_50.log" 2>&1
    done    
done
