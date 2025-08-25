#!/bin/bash

echo "--- 开始超参数网格搜索 (192 种组合, 6个任务并行) ---"

# --- 1. 定义您的参数网格 ---
learning_rates=(5e-4 1e-3 1e-5 5e-6 3e-5 6e-5 )
n_steps_list=(1000 2000)
n_epochs_list=(5 10 15 20)
ent_coefs=(0  )
batch_sizes=(256  1024 )

# --- 2. 定义计算所需和硬件相关的常量 ---
NUM_CPU_WORKERS=9
CFR_ITER_PER_STEP=2
TOTAL_CFR_ITER_TARGET=200000000
NUM_GPUS=6
LOG_PATH="logs"

# --- 3. 激活您的Conda环境 ---
. "/home/mlj/anaconda3/etc/profile.d/conda.sh" 
conda activate DDCFR_infoset

# 获取主目录路径
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
main_dir=$script_dir/..
cd "$main_dir"

# --- 4. 遍历所有参数组合并分批执行 ---
BATCH_SIZE=6
run_count=0
total_runs=$((${#learning_rates[@]} * ${#n_steps_list[@]} * ${#n_epochs_list[@]} * ${#ent_coefs[@]} * ${#batch_sizes[@]}))
log_dir="${LOG_PATH}/grid_search_logs"
mkdir -p $log_dir

for lr in "${learning_rates[@]}"; do
    for steps in "${n_steps_list[@]}"; do
        for epochs in "${n_epochs_list[@]}"; do
            for ent in "${ent_coefs[@]}"; do
                for bs in "${batch_sizes[@]}"; do

                    run_count=$((run_count+1))
                    
                    cfr_per_iteration=$((NUM_CPU_WORKERS * steps * CFR_ITER_PER_STEP))
                    total_iters_rounded=$(printf "%.0f" $(echo "$TOTAL_CFR_ITER_TARGET / $cfr_per_iteration" | bc -l))
                    
                    # ==================== 代码修改部分 START ====================
                    # 中文注释: 在日志文件名中加入 total_iters_rounded 变量
                    log_file="${log_dir}/run_${run_count}_lr_${lr}_steps_${steps}_epochs_${epochs}_ent_${ent}_bs_${bs}_iters_${total_iters_rounded}.log"
                    # ==================== 代码修改部分 END ======================
                    gpu_id=$(( (run_count - 1) % NUM_GPUS ))

                    echo ""
                    # ==================== 代码修改部分 START ====================
                    # 中文注释: 在打印信息中也加入 total_iters_rounded
                    echo "--- 提交任务 ${run_count}/${total_runs} 到 GPU ${gpu_id}: lr=${lr}, n_steps=${steps}, n_epochs=${epochs}, ent_coef=${ent}, batch_size=${bs}, total_iters=${total_iters_rounded} ---"
                    # ==================== 代码修改部分 END ======================
                    echo "--- 日志文件: ${log_file} ---"
                    
                    nohup env CUDA_VISIBLE_DEVICES=$gpu_id python -u scripts/run_ppo.py with save_log=True \
                        log_path=$LOG_PATH \
                        experiment_id=$run_count \
                        learning_rate=$lr \
                        n_steps=$steps \
                        n_epochs=$epochs \
                        ent_coef=$ent \
                        batch_size=$bs \
                        num_cpu_workers=$NUM_CPU_WORKERS \
                        total_iterations=$total_iters_rounded > "$log_file" 2>&1 &
                    
                    if (( run_count % BATCH_SIZE == 0 )); then
                        echo ""
                        echo "--- 已提交 ${BATCH_SIZE} 个任务。等待本批次完成... ---"
                        wait
                        echo "--- 本批次已完成。提交下一批次。 ---"
                    fi
                done
            done
        done
    done
done

# 等待最后一批任务全部完成
echo ""
echo "--- 所有网格搜索任务均已提交。等待最后一批完成... ---"
wait
echo ""
echo "--- 所有网格搜索任务均已成功完成。 ---"