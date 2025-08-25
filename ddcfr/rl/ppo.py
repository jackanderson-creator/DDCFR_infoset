import time
import os
import re
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


class PPO_Worker:
    """
    PPO Worker类，负责在单个设备上针对单个游戏进行数据收集和梯度计算。
    """
    def __init__(
        self,
        game_config: GameConfig,
        logger: Logger,
        seed: int,
        worker_id: int,
        log_path: str,
        experiment_id: int,
        # learning_rate: float, # 新增，为worker的本地优化器提供学习率
        n_steps: int = 256,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_range: float = 0.2,
        normalize_advantage: bool = False,
        **kwargs,
    ):
        self.worker_id = worker_id
        self.logger = logger
        set_seed(seed)

        if th.cuda.is_available():
            self.device = th.device("cuda:0")
            # self.logger.info(f"Worker {self.worker_id} 分配到 GPU: {th.cuda.get_device_name(0)}")
        else:
            self.device = th.device("cpu")
            # self.logger.info(f"Worker {self.worker_id} 使用 CPU")
        
        self.solver = DDCFRSolver(game_config, self.logger)
        
        self.obs_dim = 2
        self.action_dim = 3
        self.num_infosets = len(self.solver.states)

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_range = clip_range
        self.final_exploitability=0.0

        #使用主进程传来的学习率
        self.policy = A2CPolicy(self.obs_dim, self.action_dim, self.device, learning_rate=0)
        
        buffer_size = self.n_steps * self.num_infosets
        self.rollout_buffer = RolloutBuffer(
            buffer_size=buffer_size, obs_dim=self.obs_dim, action_dim=self.action_dim,
            device=self.device, gamma=self.gamma, gae_lambda=self.gae_lambda,
        )
        self.solver.reset_for_new_run()

            
    def learn_one_cycle(self, master_policy_state_dict: dict) -> Tuple[dict, int, str, float, int]:
        self.policy.load_state_dict(master_policy_state_dict)
        self.collect_rollouts_custom()


        # if len(self.solver.conv_history)>=500 :
        #     latest_exploitability=self.solver.conv_history[499] if self.solver.conv_history else 0.0
        # latest_exploitability = self.solver.conv_history[-1] if self.solver.conv_history else 0.0
        game_name = self.solver.game_name
        num_timesteps = self.num_infosets * self.n_steps
        final_exploitability =self.final_exploitability

        return self.rollout_buffer,num_timesteps, game_name, final_exploitability, self.worker_id

    def collect_rollouts_custom(self):
        self.rollout_buffer.reset()
        for _ in range(self.n_steps):
            obs = self.solver.get_all_infoset_states_for_ppo()
            with th.no_grad():
                actions_tensor, values_tensor, log_probs_tensor = self.policy.predict_tensors(obs)
            
            actions = actions_tensor.cpu().numpy()
            values = values_tensor.cpu().numpy().flatten()
            log_probs = log_probs_tensor.cpu().numpy()

            reward, done = self.solver.run_one_ppo_step(actions)
            
            next_obs = self.solver.get_all_infoset_states_for_ppo()
            with th.no_grad():
                next_values = self.policy.get_values(next_obs).cpu().numpy().flatten()
            
            self.rollout_buffer.add(obs, actions, reward, done, values, log_probs, next_values)

            if done:
                self.final_exploitability=self.solver.conv_history[-1]
                self.solver.reset_for_new_run()
                start_slice = self.rollout_buffer.pos - self.num_infosets
                end_slice = self.rollout_buffer.pos
                if start_slice >= 0:
                    self.rollout_buffer.next_values[start_slice:end_slice] = 0.0

        self.rollout_buffer.compute_returns_and_advantage()



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
    
    def predict_deterministic(self, obs_np: np.ndarray) -> np.ndarray:
        obs = th.as_tensor(obs_np, device=self.device)
        with th.no_grad():
            action_mean, _ = self.ac_net(obs)
        return action_mean.cpu().numpy()

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
        self.feature_model = nn.Sequential(nn.Linear(input_dim, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
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
        self.buffer_size = int(buffer_size)
        self.obs_dim, self.action_dim, self.device, self.gamma, self.gae_lambda = obs_dim, action_dim, device, gamma, gae_lambda
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float64)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float64)
        self.rewards, self.dones, self.values, self.log_probs, self.next_values, self.advs, self.rets = (np.zeros(self.buffer_size, dtype=np.float64) for _ in range(7))
        self.pos, self.full = 0, False

    def add(self, obs, actions, reward, done, values, log_probs, next_values):
        batch_size = obs.shape[0]
        if self.pos + batch_size > self.buffer_size:
            print(f"Warning: RolloutBuffer overflow detected. Buffer size: {self.buffer_size}, current pos: {self.pos}, incoming batch size: {batch_size}. Ignoring this batch.")
            self.full = True
            return
        end_pos = self.pos + batch_size
        self.obs[self.pos:end_pos], self.actions[self.pos:end_pos], self.values[self.pos:end_pos], self.log_probs[self.pos:end_pos], self.next_values[self.pos:end_pos] = obs, actions, values, log_probs, next_values
        self.rewards[self.pos:end_pos], self.dones[self.pos:end_pos] = reward, float(done)
        self.pos = end_pos
        if self.pos >= self.buffer_size: self.full = True

    def compute_returns_and_advantage(self):
        last_gae = 0
        for step in reversed(range(self.pos)):
            delta = self.rewards[step] + self.gamma * self.next_values[step] * (1 - self.dones[step]) - self.values[step]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * last_gae
            self.advs[step] = last_gae
        self.rets[:self.pos] = self.advs[:self.pos] + self.values[:self.pos]

    def get(self, batch_size: int):
        if self.pos < batch_size: return
        indices = np.random.permutation(self.pos)
        for start_idx in range(0, self.pos, batch_size):
            end_idx = start_idx + batch_size
            if end_idx > self.pos: continue
            batch_indices = indices[start_idx : end_idx]
            data = (self.obs[batch_indices], self.actions[batch_indices], self.values[batch_indices], self.log_probs[batch_indices], self.advs[batch_indices], self.rets[batch_indices])
            yield RolloutBufferSamples(*tuple(map(lambda x: th.as_tensor(x, device=self.device).double(), data)))