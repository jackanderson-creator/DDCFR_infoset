import os
import re
import time
import torch as th
import numpy as np
import torch.multiprocessing as mp
from sacred import Experiment
from sacred.observers import FileStorageObserver
from collections import OrderedDict
import torch.nn.functional as F

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
    learning_rate = 1e-3
    n_steps = 500
    n_epochs = 10
    total_iterations = 5000
    num_cpu_workers = 20
    num_train_games = 4
    batch_size = 500
    gamma = 0.99
    gae_lambda = 1.0
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    clip_range = 0.2
    normalize_advantage = True
    seed = 0
    save_log = True
    save_interval = 10



# # 新增一个函数，专门用于在主进程中进行训练
# def train_on_master(policy, all_rollout_buffers, n_epochs, batch_size, vf_coef, ent_coef, max_grad_norm, clip_range, normalize_advantage):
#     """
#     使用从所有worker收集到的数据在主进程中训练模型。
#     """
#     policy.train()
    
#     # 初始化用于累加损失的变量
#     total_policy_loss = 0.0
#     total_value_loss = 0.0
#     total_entropy_loss = 0.0
#     num_batches = 0


#     for epoch in range(n_epochs):
#         # 将所有buffer中的数据合并起来进行训练
#         for rollout_buffer in all_rollout_buffers:
#             for data in rollout_buffer.get(batch_size):
#                 values, log_probs, entropy = policy.evaluate_actions(data.obs, data.actions)
#                 advs = data.advs
#                 if normalize_advantage and len(advs) > 1:
#                     advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                
#                 ratio = th.exp(log_probs - data.old_log_probs)
#                 policy_loss = -th.min(advs * ratio, advs * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
#                 value_loss = F.mse_loss(data.rets, values)
#                 entropy_loss = -th.mean(entropy)
#                 loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                
#                  # 累加各个损失值
#                 total_policy_loss += policy_loss.item()
#                 total_value_loss += value_loss.item()
#                 total_entropy_loss += entropy_loss.item()
#                 num_batches += 1


#                 policy.optimizer.zero_grad()
#                 loss.backward()
#                 th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
#                 policy.optimizer.step()
#                 break

#     # 计算平均损失
#     avg_losses = {
#         "policy_loss": total_policy_loss / num_batches if num_batches > 0 else 0.0,
#         "value_loss": total_value_loss / num_batches if num_batches > 0 else 0.0,
#         "entropy_loss": total_entropy_loss / num_batches if num_batches > 0 else 0.0,
#     }

#     return avg_losses




def train_on_master(policy, all_rollout_buffers, n_epochs, batch_size, n_steps, vf_coef, ent_coef, max_grad_norm, clip_range, normalize_advantage):
    """
    只从每个buffer中提取第一个信息集的数据，合并、洗牌后进行训练。
    """
    policy.train()
    
    if not all_rollout_buffers or all_rollout_buffers[0].pos == 0:
        return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0}

    # 创建列表，用于存放从每个buffer中提取出的、只属于第一个信息集的数据
    infoset0_obs, infoset0_actions, infoset0_old_values, infoset0_old_log_probs, infoset0_advantages, infoset0_returns = [], [], [], [], [], []
    # i=1
    for buf in all_rollout_buffers:
        if buf.pos == 0: continue
        
        # 从buffer的填充位置和n_steps推断出信息集的数量
        # 注意：这里假设buf.pos是n_steps的整数倍，这在我们的代码中是成立的
        num_infosets = buf.pos // n_steps
        if num_infosets == 0: continue


        # 使用切片技术 [0::num_infosets] 来提取属于第一个信息集的数据
        # 它的意思是：从索引0开始，每隔 num_infosets 个位置取一个元素
        infoset0_obs.append(buf.obs[:buf.pos][0::num_infosets])
        infoset0_actions.append(buf.actions[:buf.pos][0::num_infosets])
        infoset0_old_values.append(buf.values[:buf.pos][0::num_infosets])
        infoset0_old_log_probs.append(buf.log_probs[:buf.pos][0::num_infosets])
        infoset0_advantages.append(buf.advs[:buf.pos][0::num_infosets])
        infoset0_returns.append(buf.rets[:buf.pos][0::num_infosets])
        # print(f"第{i}个游戏的buffer中取出的obs的维度是{buf.obs[:buf.pos][0::num_infosets].shape}/n")
        # i+=1
    # 如果没有收集到任何数据，则直接返回
    if not infoset0_obs:
        return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0}

    # 将所有游戏的“第一个信息集”数据合并成一个大的Numpy数组
    full_obs = np.concatenate(infoset0_obs)
    full_actions = np.concatenate(infoset0_actions)
    full_old_values = np.concatenate(infoset0_old_values)
    full_old_log_probs = np.concatenate(infoset0_old_log_probs)
    full_advantages = np.concatenate(infoset0_advantages)
    full_returns = np.concatenate(infoset0_returns)
    
    # print(f"我们的obs的维度{full_obs.shape}/n")
    # # print(f"values的维度{full_old_values.shape}/n")
    # print(f"我们的advs的维度{full_advantages.shape}/n")


    dataset_size = full_obs.shape[0]

    # 初始化用于累加损失的变量
    total_policy_loss, total_value_loss, total_entropy_loss = 0.0, 0.0, 0.0
    num_batches = 0


    for epoch in range(n_epochs):
        indices = np.random.permutation(dataset_size)

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = start_idx + batch_size
            if end_idx > dataset_size: continue
            
            batch_indices = indices[start_idx:end_idx]
            device = policy.device
            
            # --- PPO 更新
            obs_tensor = th.as_tensor(full_obs[batch_indices], device=device).double()
            actions_tensor = th.as_tensor(full_actions[batch_indices], device=device).double()
            old_log_probs_tensor = th.as_tensor(full_old_log_probs[batch_indices], device=device).double()
            advantages_tensor = th.as_tensor(full_advantages[batch_indices], device=device).double()
            returns_tensor = th.as_tensor(full_returns[batch_indices], device=device).double()

            values, log_probs, entropy = policy.evaluate_actions(obs_tensor, actions_tensor)
            advs = advantages_tensor
            if normalize_advantage and len(advs) > 1:
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            
            ratio = th.exp(log_probs - old_log_probs_tensor)
            policy_loss = -th.min(advs * ratio, advs * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
            value_loss = F.mse_loss(returns_tensor, values)
            entropy_loss = -th.mean(entropy)
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            total_policy_loss += policy_loss.item(); total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item(); num_batches += 1
            
            policy.optimizer.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm); policy.optimizer.step()

    avg_losses = {
        "policy_loss": total_policy_loss / num_batches if num_batches > 0 else 0.0,
        "value_loss": total_value_loss / num_batches if num_batches > 0 else 0.0,
        "entropy_loss": total_entropy_loss / num_batches if num_batches > 0 else 0.0,
    }
    return avg_losses






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
            rollout_buffer,timesteps, game_name, exploitability, worker_id = worker.learn_one_cycle(master_policy_state_dict)
            
            # 将结果放入结果队列
            result_queue.put({
                "rollout_buffer": rollout_buffer,
                "timesteps": timesteps, 
                "game_name": game_name,
                "exploitability": exploitability, 
                "worker_id": worker_id,
            })

        except Exception as e:
            import traceback
            result_queue.put({"error": str(e), "traceback": traceback.format_exc(), "worker_id": worker_configs.get("worker_id", -1)})


# --- 主函数 (已重构为持久化工作池模式) ---
@ex.automain
def main(_run, seed, save_log, log_path, experiment_id, num_cpu_workers, num_train_games, total_iterations, save_interval, learning_rate, n_epochs, batch_size,n_steps,  vf_coef, ent_coef, max_grad_norm, clip_range, normalize_advantage, **kwargs):
    full_log_path = os.path.join(log_path, str(experiment_id))
    logger = Logger(folder=full_log_path, name=str(experiment_id), writer_strings=["stdout", "csv", "tensorboard"])
    logger.info("================ 主进程开始 ( (中央集权训练模模式) ================")

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
    master_policy = A2CPolicy(obs_dim=2, action_dim=3, device=device, learning_rate=learning_rate)
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
    
    NUM_GPUS = 6
    logger.info(f"正在启动 {num_cpu_workers} 个持久化工作进程...")
    for i in range(num_cpu_workers):
        gpu_id = i % NUM_GPUS
        worker_configs = {
            "seed": seed + i, "worker_id": i, "log_path": log_path,
            "experiment_id": experiment_id,"gpu_id": gpu_id,
            "learning_rate": learning_rate,"n_steps":n_steps,"batch_size":batch_size, **kwargs
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

        # ==================== 代码修改部分 START ====================
        #  收集所有worker返回的rollout_buffer
        results = []; all_rollout_buffers = []
        for _ in range(num_cpu_workers):
            res = result_queue.get()
            if "error" in res:
                logger.error(f"工作进程 {res['worker_id']} 发生错误: {res['error']}")
                if "traceback" in res: logger.error(f"详细错误追踪:\n{res['traceback']}")
                continue
            results.append(res)
            all_rollout_buffers.append(res["rollout_buffer"])
        

        # j=1
        # print(f"all_rollout_buffers的维度{len(all_rollout_buffers)}!!!!!!!!!!!!!!!!!/n")
        # print(f"所有buffer的内容：{all_rollout_buffers}/n")
        # for buf in all_rollout_buffers:
        #     d= np.array(buf.obs)
        #     print(f"all_rollout_buffers的第{j}个obs维度{d.shape}!!!!!!!!!!!!!!!!!/n")
        #     j+=1






        if not results:
            logger.error("所有工作进程都返回了错误，训练终止。"); break

        # 在主进程中进行训练
        avg_losses=train_on_master(master_policy, all_rollout_buffers, n_epochs, batch_size,n_steps, vf_coef, ent_coef, max_grad_norm, clip_range, normalize_advantage)
        logger.info("主模型已在所有worker数据上完成更新。")
        # ==================== 代码修改部分 END ======================
        
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

        # 中文注释: 记录全局的平均损失值
        logger.record("loss/policy_loss", avg_losses["policy_loss"])
        logger.record("loss/value_loss", avg_losses["value_loss"])
        logger.record("loss/entropy_loss", avg_losses["entropy_loss"])
        
        logger.dump(step=total_timesteps)

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