import time
import os
from collections import deque
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ddcfr.cfr.ddcfr import DDCFRSolver
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger
from ddcfr.utils.utils import set_seed


class PPO:
    """
    PPO main class for direct interaction with DDCFRSolver.
    Handles multiple games by training on them sequentially.
    """
    def __init__(
        self,
        train_game_configs: List[GameConfig],
        logger: Logger,
        learning_rate: float = 0.001,
        n_steps: int = 256,
        batch_size: int = 256,
        n_epochs: int = 20,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_range: float = 0.2,
        normalize_advantage: bool = False,
        seed: int = 0,
        log_interval: int = 1,
        save_interval: int = 100,
        save_path: Optional[str] = None,
        device: Optional[th.device] = None,
    ):
        self.logger = logger
        self.device = device if device else th.device("cuda" if th.cuda.is_available() else "cpu")
        set_seed(seed)
        
        # 创建所有游戏的Solver实例
        self.game_configs = train_game_configs
        self.solvers = {cfg.name: DDCFRSolver(cfg, self.logger) for cfg in self.game_configs}
        self.game_names = [cfg.name for cfg in self.game_configs]
        self.num_games = len(self.game_names)
        
        # 确定缓冲区大小，基于所有游戏中信息集数量的最大值
        max_num_infosets = 0
        for solver in self.solvers.values():
            if len(solver.states) > max_num_infosets:
                max_num_infosets = len(solver.states)

        self.obs_dim = 3
        self.action_dim = 3

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_range = clip_range

        # 初始化PPO策略网络
        self.policy = A2CPolicy(self.obs_dim, self.action_dim, self.device, self.learning_rate)
        
        # RolloutBuffer的大小基于最大的游戏
        buffer_size = self.n_steps * max_num_infosets
        self.rollout_buffer = RolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # 设置模型保存路径
        self.save_path = save_path
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

    def learn(self, total_timesteps: int):
        self.start_time = time.time()
        self.num_timesteps = 0
        self.iterations = 0
        current_game_idx = 0

        # 初始化所有solver
        for solver in self.solvers.values():
            solver.reset_for_new_run()

        while self.num_timesteps < total_timesteps:
            # 顺序选择当前要训练的游戏
            game_name = self.game_names[current_game_idx]
            active_solver = self.solvers[game_name]
            
            self.logger.record("train/current_game_name", game_name)

            # 使用当前选择的solver收集数据
            self.collect_rollouts_custom(active_solver, n_steps=self.n_steps)
            
            # 训练PPO策略
            self.train()

            self.iterations += 1
            if self.log_interval > 0 and self.iterations % self.log_interval == 0:
                self.dump_logs()
            
            if self.save_path and self.save_interval > 0 and self.iterations % self.save_interval == 0:
                self.save_model()
            
            # 切换到下一个游戏
            current_game_idx = (current_game_idx + 1) % self.num_games

    def collect_rollouts_custom(self, active_solver: DDCFRSolver, n_steps: int):
        self.rollout_buffer.reset()
        num_infosets_current_game = len(active_solver.states)
        
        for _ in range(n_steps):
            obs = active_solver.get_all_infoset_states_for_ppo()
            
            with th.no_grad():
                actions_tensor, values_tensor, log_probs_tensor = self.policy.predict_tensors(obs)
            
            actions = actions_tensor.cpu().numpy()
            values = values_tensor.cpu().numpy().flatten()
            log_probs = log_probs_tensor.cpu().numpy()

            reward, done = active_solver.run_one_ppo_step(actions)
            self.num_timesteps += num_infosets_current_game
            
            next_obs = active_solver.get_all_infoset_states_for_ppo()
            with th.no_grad():
                next_values = self.policy.get_values(next_obs).cpu().numpy().flatten()
            
            self.rollout_buffer.add(obs, actions, reward, done, values, log_probs, next_values)

            if done:
                active_solver.reset_for_new_run()
                start_slice = self.rollout_buffer.pos - num_infosets_current_game
                end_slice = self.rollout_buffer.pos
                if start_slice >= 0:
                    self.rollout_buffer.next_values[start_slice:end_slice] = 0.0

        self.rollout_buffer.compute_returns_and_advantage()

    def train(self):
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                values, log_probs, entropy = self.policy.evaluate_actions(data.obs, data.actions)
                
                advs = data.advs
                if self.normalize_advantage and len(advs) > 1:
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                ratio = th.exp(log_probs - data.old_log_probs)
                policy_loss1 = advs * ratio
                policy_loss2 = advs * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss1, policy_loss2).mean()
                
                value_loss = F.mse_loss(data.rets, values)
                entropy_loss = -th.mean(entropy)
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
        
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())

    def save_model(self):
        """Saves the policy network's state dictionary."""
        model_path = os.path.join(self.save_path, f"model_{self.num_timesteps}_steps.pth")
        th.save(self.policy.state_dict(), model_path)
        self.logger.log(f"Saving model to {model_path}")

    def dump_logs(self):
        # **最终修复**: 在记录任何新数据之前，立即清空logger的缓存。
        # 这确保了本次dump操作只包含下面明确记录的数据。
        if hasattr(self.logger, 'name_to_value'):
            self.logger.name_to_value.clear()
        time_elasped = time.time() - self.start_time
        fps = self.num_timesteps / time_elasped if time_elasped > 0 else 0
        self.logger.record("time/num_timesteps", self.num_timesteps)
        self.logger.record("time/iterations", self.iterations)
        self.logger.record("time/fps", fps)
        for name, solver in self.solvers.items():
            if solver.conv_history:
                 self.logger.record(f"exploitability/{name}", solver.conv_history[-1])
        self.logger.dump(step=self.num_timesteps)
       


class A2CPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, device, learning_rate):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.ac_net = ACNetwork(obs_dim, action_dim).to(th.double).to(device)
        self.log_std = nn.Parameter(th.zeros(action_dim, device=device), requires_grad=True)
        self.optimizer = th.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    def predict_tensors(self, obs_np: np.ndarray) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = th.as_tensor(obs_np, device=self.device)
        action_mean, value = self.ac_net(obs)
        std = self.log_std.exp()
        dist = th.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, value, log_prob
    
    def get_values(self, obs_np: np.ndarray) -> th.Tensor:
        obs = th.as_tensor(obs_np, device=self.device)
        _ , value = self.ac_net(obs)
        return value

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        action_mean, value = self.ac_net(obs)
        std = self.log_std.exp()
        dist = th.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value.flatten(), log_prob, entropy

class ACNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.feature_model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU()
        )
        self.policy_head = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs):
        feature = self.feature_model(obs)
        action_mean = self.policy_head(feature)
        value = self.value_head(feature)
        return action_mean, value

class RolloutBufferSamples(NamedTuple):
    obs: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_probs: th.Tensor
    advs: th.Tensor
    rets: th.Tensor

class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, device, gamma, gae_lambda):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float64)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float64)
        self.dones = np.zeros(self.buffer_size, dtype=np.float64)
        self.values = np.zeros(self.buffer_size, dtype=np.float64)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float64)
        self.next_values = np.zeros(self.buffer_size, dtype=np.float64)
        self.advs = np.zeros(self.buffer_size, dtype=np.float64)
        self.rets = np.zeros(self.buffer_size, dtype=np.float64)
        self.pos = 0
        self.full = False

    def add(self, obs, actions, reward, done, values, log_probs, next_values):
        batch_size = obs.shape[0]
        if self.pos + batch_size > self.buffer_size:
            self.full = True
            batch_size = self.buffer_size - self.pos
            if batch_size <= 0: return
            obs, actions, values, log_probs, next_values = (d[:batch_size] for d in [obs, actions, values, log_probs, next_values])

        end_pos = self.pos + batch_size
        self.obs[self.pos:end_pos] = obs
        self.actions[self.pos:end_pos] = actions
        self.values[self.pos:end_pos] = values
        self.log_probs[self.pos:end_pos] = log_probs
        self.next_values[self.pos:end_pos] = next_values
        self.rewards[self.pos:end_pos] = reward
        self.dones[self.pos:end_pos] = float(done)
        
        self.pos = end_pos
        if self.pos >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self):
        last_gae = 0
        end_pos = self.pos
        for step in reversed(range(end_pos)):
            delta = self.rewards[step] + self.gamma * self.next_values[step] * (1 - self.dones[step]) - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * last_gae
            self.advs[step] = last_gae
        self.rets[:end_pos] = self.advs[:end_pos] + self.values[:end_pos]

    def get(self, batch_size: int):
        indices = np.random.permutation(self.pos)
        for start_idx in range(0, self.pos, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            data = (
                self.obs[batch_indices],
                self.actions[batch_indices],
                self.values[batch_indices],
                self.log_probs[batch_indices],
                self.advs[batch_indices],
                self.rets[batch_indices],
            )
            yield RolloutBufferSamples(*tuple(map(lambda x: th.as_tensor(x, device=self.device).double(), data)))