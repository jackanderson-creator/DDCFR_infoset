import os
import re
import time
import torch as th
import numpy as np
import torch.multiprocessing as mp
from sacred import Experiment
from sacred.observers import FileStorageObserver
from collections import OrderedDict

# 导入您项目中的相关模块
from ddcfr.utils.logger import Logger
from ddcfr.game.game_config import get_train_configs
from ddcfr.rl.ppo import PPO_Worker, A2CPolicy

# --- Sacred 实验设置 ---
ex = Experiment("ddcfr_ppo")
# ex.observers.append(FileStorageObserver('sacred_runs'))

# --- 实验参数配置 ---
@ex.config
def default_config():
    log_path = "logs"
    experiment_id = 1
    learning_rate = 1e-4
    n_steps = 1024
    n_epochs = 10
    total_iterations = 10000
    num_cpu_workers = 4
    num_train_games = 4
    batch_size = 256
    gamma = 0.99
    gae_lambda = 1.0
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    clip_range = 0.2
    normalize_advantage = True
    seed = 0
    save_log = True
    save_interval = 30

# --- 工作进程执行的函数 (持久化模式) ---
def worker_fn(task_queue, result_queue, game_config, worker_configs):
    """
    每个独立的worker进程运行此函数。
    它会先初始化一次，然后进入一个无限循环等待任务。
    """
    # 1. 在进程开始时，只进行一次初始化
    worker_log_path = os.path.join(worker_configs['log_path'], str(worker_configs['experiment_id']))
    worker_logger = Logger(
        folder=worker_log_path, name=f"worker_{worker_configs['worker_id']}", writer_strings=["stdout"]
    )
    worker_configs['logger'] = worker_logger
    
    # 创建 PPO_Worker 实例一次
    worker = PPO_Worker(game_config=game_config, **worker_configs)

    # 2. 进入主循环，等待任务
    while True:
        try:
            # 从任务队列获取任务，这里会阻塞直到有任务为止
            task = task_queue.get()

            # 收到关闭信号，则退出循环
            if task == 'shutdown':
                break
            
            # 正常任务是主模型的参数字典
            master_policy_state_dict = task
            
            # 执行一轮学习
            updated_state_dict, timesteps, game_name, exploitability, worker_id = worker.learn_one_cycle(master_policy_state_dict)
            
            # 将结果放入结果队列
            result_queue.put({
                "state_dict": updated_state_dict, "timesteps": timesteps, "game_name": game_name,
                "exploitability": exploitability, "worker_id": worker_id,
            })

        except Exception as e:
            import traceback
            result_queue.put({"error": str(e), "traceback": traceback.format_exc(), "worker_id": worker_configs.get("worker_id", -1)})


# --- 主函数 (已重构为持久化工作池模式) ---
@ex.automain
def main(_run, seed, save_log, log_path, experiment_id, num_cpu_workers, num_train_games, total_iterations, save_interval, learning_rate, **kwargs):
    full_log_path = os.path.join(log_path, str(experiment_id))
    logger = Logger(folder=full_log_path, name=str(experiment_id), writer_strings=["stdout", "csv", "tensorboard"])
    logger.info("================ 主进程开始 (持久化工作池模式) ================")

    kwargs.pop('local_reward_weight', None)

    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    model_save_dir = os.path.join(log_path, str(experiment_id))
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info(f"模型将保存在: {model_save_dir}")

    train_games = get_train_configs(num_train_games=num_cpu_workers)
    # for config in train_games:
    #     config.iterations = float('inf')

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    master_policy = A2CPolicy(obs_dim=4, action_dim=3, device=device, learning_rate=0)
    logger.info(f"主策略网络已在 {device} 上初始化。")
    
    # 尝试加载模型
    try:
        model_files = [f for f in os.listdir(model_save_dir) if re.match(r"model_\d+_iterations\.pth", f)]
        if model_files:
            latest_iter = -1
            latest_model_file = None
            for f in model_files:
                iter_num = int(re.search(r"model_(\d+)_iterations\.pth", f).group(1))
                if iter_num > latest_iter:
                    latest_iter = iter_num
                    latest_model_file = f
            
            if latest_model_file:
                model_path = os.path.join(model_save_dir, latest_model_file)
                logger.info(f"找到最新模型: {model_path}，正在加载...")
                master_policy.load_state_dict(th.load(model_path, map_location=device))
                logger.info("模型加载成功！")
    except Exception as e:
                logger.error(f"加载模型时出错: {e}。将从头开始训练。")

    # ==================== 代码修改部分 START ====================
    # 4. 创建任务队列、结果队列，并一次性启动所有工作进程
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    workers = []
    
    logger.info(f"正在启动 {num_cpu_workers} 个持久化工作进程...")
    for i in range(num_cpu_workers):
        worker_configs = {
            "seed": seed + i, "worker_id": i, "log_path": log_path,
            "experiment_id": experiment_id, "learning_rate": learning_rate, **kwargs
        }
        game_config = train_games[i % len(train_games)]
        
        p = mp.Process(target=worker_fn, args=(task_queue, result_queue, game_config, worker_configs))
        workers.append(p)
        p.start()
        logger.info(f"工作进程 {i} 已启动，负责游戏: {game_config.name}")

    # 5. 主训练循环
    logger.info("================ 开始主训练循环 ================")
    total_timesteps = 0
    for iteration in range(1, total_iterations + 1):
        iter_start_time = time.time()
        
        # 将当前模型参数作为任务分发给所有worker
        master_policy_state_dict_cpu = {k: v.cpu() for k, v in master_policy.state_dict().items()}
        for _ in range(num_cpu_workers):
            task_queue.put(master_policy_state_dict_cpu)

        # 收集所有worker返回的结果
        results = []
        for _ in range(num_cpu_workers):
            res = result_queue.get()
            if "error" in res:
                logger.error(f"工作进程 {res['worker_id']} 发生错误: {res['error']}")
                if "traceback" in res: logger.error(f"详细错误追踪:\n{res['traceback']}")
                continue
            results.append(res)
        
        if not results:
            logger.error("所有工作进程都返回了错误，训练终止。")
            break

        # 平均模型参数
        avg_state_dict = results[0]["state_dict"]
        for res in results[1:]:
            for key in avg_state_dict:
                avg_state_dict[key] += res["state_dict"][key]
        num_successful_workers = len(results)
        for key in avg_state_dict:
            avg_state_dict[key] /= num_successful_workers
        master_policy.load_state_dict(avg_state_dict)
        
        # 日志记录
        # ... (日志记录逻辑保持不变) ...
        iter_time = time.time() - iter_start_time
        current_timesteps = sum([res["timesteps"] for res in results])
        total_timesteps += current_timesteps
        logger.info(f"--- Iteration {iteration}/{total_iterations} ---")
        logger.record("info/iteration", iteration)
        logger.record("info/total_timesteps", total_timesteps)
        logger.record("perf/iteration_time", iter_time)
        for res in results:
            game_log_name = f"exploitability/{res['game_name']}"
            logger.record(game_log_name, res["exploitability"])
        logger.dump(step=iteration)

        # 保存模型
        if iteration % save_interval == 0:
            save_path = os.path.join(model_save_dir, f"model_{iteration}_iterations.pth")
            th.save(master_policy.state_dict(), save_path)
            logger.info(f"模型已保存到: {save_path}")

    # 7. 训练结束，发送关闭信号并清理工作进程
    logger.info("================ 训练结束，正在清理进程 ================")
    for _ in range(num_cpu_workers):
        task_queue.put('shutdown')
    
    for p in workers:
        p.join() # 等待所有worker进程正常退出
    logger.info("所有工作进程已清理。")
    # ==================== 代码修改部分 END ======================