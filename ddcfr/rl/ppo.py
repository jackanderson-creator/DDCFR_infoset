import time
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Tuple

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# from ddcfr.cfr.cfr_env import make_cfr_vec_env
from ddcfr.cfr.cfr_env import make_multi_cfr_env
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger
from ddcfr.utils.utils import set_seed


class PPO:
    def __init__(
        self,
        train_game_configs: List[GameConfig],
        logger: Logger,
        learning_rate: float = 0.001,
        n_steps: int = 256,
        batch_size: int = 256,
        n_epochs: int = 20,
        n_envs: int = 2,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_range: float = 0.2,
        clip_range_vf: float = 0.0,
        normalize_advantage: bool = False,
        seed: int = 0,
        log_interval: int = 0,
        device: Optional[th.device] = None,
    ):
        self.train_game_configs = train_game_configs
        self.n_envs = n_envs
        self.env = self.make_env(train_game_configs, n_envs)
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.seed = seed
        self.log_interval = log_interval
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.logger = logger
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.rollout_buffers=RolloutBuffers(self.device,self.batch_size,self.n_envs)
        if self.device is None:
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.setup()

    def setup(self):
        self.set_seed(self.seed)
        self.policy = A2CPolicy(
            device=self.device,
            learning_rate=self.learning_rate,
        )


    def setup_learn(self, total_timesteps: int):
        self.start_time = time.time()
        self.total_timesteps = total_timesteps
        self.num_timesteps = 0
        self.num_episodes = 0
        # self.last_obs = self.env.reset()
        self.n_updates = 0
        self.ep_info_buffer = deque(maxlen=10)

    def learn(self, total_timesteps: int):
        self.setup_learn(total_timesteps)
        self.iterations = 0
        while self.num_timesteps < total_timesteps:
            self.rollout_buffers.reset()
            self.collect_rollouts(
                env=self.env,
                n_steps=self.n_steps
            )
            self.iterations += 1
            if self.log_interval > 0 and self.iterations % self.log_interval == 0:
                self.dump_logs()
            self.train()

    def setup_buffer(self,num_info_sets):
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            num_info_sets=num_info_sets,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def collect_rollouts(
        self, env: gym.Env, n_steps: int
    ):
        # rollout_buffer.reset()
        #20个游戏环境一个个来
        for i in range(self.n_envs):
            num_collected_steps = 0
            self.setup_buffer(self.env[i].num_info_sets)
            self.last_obs=self.env[i].reset()
            rollout_buffer=self.rollout_buffer
            while num_collected_steps < n_steps:
                acts, values, log_probs = self.policy.predict(
                    self.last_obs
                )
                new_obs, reward, done, info = env[i].step(acts)
                #现在dones只是一个环境的，不需要遍历
                # for idx, done in enumerate(dones):


                if done and info.get("TimeLimit.truncated", False):
                    terminal_obs = info["terminal_observation"]
                    _, terminal_value, _ = self.policy.predict(terminal_obs)
                    terminal_value=np.mean(terminal_value)
                    reward = reward + self.gamma * terminal_value

                num_collected_steps += 1
                self.update_info_buffer(done, info)

                acts_abg=acts
                self.store_transition(
                    rollout_buffer, acts_abg, reward, done, values, log_probs
                )
                self.last_obs = new_obs

                if done:
                    self.num_episodes += 1
            self.num_timesteps += n_steps
            _, eventual_value, _ = self.policy.predict(new_obs)
            rollout_buffer.compute_returns_and_advantage(eventual_value=eventual_value)
            self.rollout_buffers.add(rollout_buffer)
    def train(self):
        entropy_losses = []
        value_losses = []
        policy_losses = []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            for data in self.rollout_buffers.get(self.batch_size):
                data = RolloutBufferSamples(
                    obs=data.obs.to(self.device),
                    acts_abg=data.acts_abg.to(self.device),
                    old_values=data.old_values.to(self.device),
                    old_log_probs=data.old_log_probs.to(self.device),
                    advs=data.advs.to(self.device),
                    rets=data.rets.to(self.device)
                )
                values, log_probs, entropy = self.policy.evaluate(
                    data.obs, data.acts_abg
                )
                advs = data.advs
                if self.normalize_advantage:
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                ratio = th.exp(log_probs - data.old_log_probs)
                policy_loss1 = advs * ratio
                policy_loss2 = advs * th.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -th.min(policy_loss1, policy_loss2).mean()
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > self.clip_range).float()
                ).item()
                value_loss = F.mse_loss(data.rets, values)
                entropy_loss = -th.mean(entropy)
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.policy.optimizer.zero_grad()
                loss.backward()

                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)

        self.n_updates += self.n_epochs
        self.logger.record("train/n_updates", self.n_updates)
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))

    def make_env(self, game_configs: List[GameConfig], n_envs: int) -> gym.Env:
        # env = make_cfr_vec_env(game_configs, n_envs=n_envs)
        #创建环境列表
        env=[]
        for i in range(n_envs):
            # game_id=i%len(game_configs)
            single_env=make_multi_cfr_env(game_configs,i)
            env.append(single_env)
        return env

    # def set_seed(self, seed: int):
    #     set_seed(seed)
    #     self.env.seed(seed)
    #     self.env.action_space.seed(seed)

    #现在env是个列表，不能直接seed，要一个个
    def set_seed(self, seed: int):
        set_seed(seed)
        for i, env in enumerate(self.env):
            env.seed(seed + i)
            env.action_space.seed(seed + i)

    def store_transition(
        self,
        rollout_buffer: "RolloutBuffer",
        acts_abg: np.ndarray,
        reward ,
        done,
        values: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        rollout_buffer.add(
            self.last_obs, acts_abg,reward, done, values, log_probs
        )

    def update_info_buffer(
        self,
        done: List[bool],
        info: List[dict],
    ) -> None:
        # for done, info in zip(dones, infos):
        if done:
            self.ep_info_buffer.append(info)

    def dump_logs(self) -> None:
        time_elasped = time.time() - self.start_time
        fps = self.num_timesteps / time_elasped
        if len(self.ep_info_buffer) > 0:
            self.logger.record(
                "rollout/conv_mean",
                np.mean([ep_info["conv"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/returns_mean",
                np.mean([ep_info["returns"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("train/learning_rate", self.learning_rate)
        self.logger.record("time/num_episodes", self.num_episodes)
        self.logger.record("time/num_timesteps", self.num_timesteps)
        self.logger.record("time/time_elasped", time_elasped)
        self.logger.record("time/fps", fps)
        self.logger.dump(step=self.num_timesteps)
        self.logger.record("cfr_model", self.policy)
        self.logger.dump(step=self.iterations)


class A2CPolicy(nn.Module):
    def __init__(
        self,
        device: th.device,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.build_net()

    def build_net(self):
        self.ac_net = self.make_ac_net()
        self.optimizer = th.optim.Adam(
            params=self.parameters(), lr=self.learning_rate, eps=1e-5
        )

    def make_ac_net(self) -> "ACNetwork":
        obs_dim = 4
        abg_dim = 3
        # tau_dim = self.action_space["tau"].n
        self.abg_log_std = nn.Parameter(
            th.zeros(abg_dim, device=self.device), requires_grad=True
        )
        return ACNetwork(obs_dim, abg_dim).to(th.double).to(self.device)

    def predict(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = self.obs_to_tensor(obs)
        with th.no_grad():
            abg_logits,  value = self.ac_net(obs)

            abg_std = self.abg_log_std.exp()
            abg_pi = th.distributions.Normal(abg_logits, abg_std)
            abg = abg_pi.sample()
            abg_log_prob = abg_pi.log_prob(abg)

            # tau_pi = th.distributions.Categorical(logits=tau_logits)
            # tau = tau_pi.sample()
            # tau_log_prob = tau_pi.log_prob(tau)

            # log_prob = abg_log_prob.sum(dim=1) + tau_log_prob
            log_prob = abg_log_prob.sum(dim=1)
        abg = abg.cpu().numpy()
        # tau = tau.cpu().numpy()
        value = value.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        # act = self.make_action(abg, tau)

        return abg, value, log_prob

    def forward(self, obs: np.ndarray) -> np.ndarray:
        obs = self.obs_to_tensor(obs)
        with th.no_grad():
            #因为测试时我们的输入不再是单个，所以不用reshape
            # obs = obs.reshape(1, -1)
            abg,  _ = self.ac_net(obs)
            # tau = th.argmax(tau_logits, dim=1, keepdim=True)
        abg = abg.cpu().numpy()
        # tau = tau.cpu().numpy()[0][0]
        # act = dict(abg=abg, tau=tau)
        act=abg
        return act

    # def make_action(
    #     self, abgs: np.ndarray, taus: np.ndarray
    # ) -> List[Dict[str, np.ndarray]]:
    #     acts = []
    #     for abg, tau in zip(abgs, taus):
    #         act = dict(abg=abg, tau=tau)
    #         acts.append(act)
    #     return acts

    def evaluate(
        self, obs: th.Tensor, abg: th.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = obs.to(self.device)
        abg = abg.to(self.device)
        abg_logits, value = self.ac_net(obs)

        abg_std = self.abg_log_std.exp()
        abg_pi = th.distributions.Normal(abg_logits, abg_std)
        abg_log_prob = abg_pi.log_prob(abg)
        abg_entropy = abg_pi.entropy()

        # tau_pi = th.distributions.Categorical(logits=tau_logits)
        # tau = tau_pi.sample()
        # tau_log_prob = tau_pi.log_prob(tau)
        # tau_entropy = tau_pi.entropy()

        # log_prob = abg_log_prob.sum(dim=1) + tau_log_prob
        # entropy = abg_entropy.sum(dim=1) + tau_entropy
        log_prob = abg_log_prob.sum(dim=1)
        entropy = abg_entropy.sum(dim=1)
        return value, log_prob, entropy

    def obs_to_tensor(self, obs: np.ndarray) -> th.Tensor:
        return th.tensor(obs, dtype=th.double).to(self.device)

    def load(self, param_path):
        self.load_state_dict(th.load(param_path))


class ACNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        abg_dim: int
    ):
        super().__init__()
        self.feature_model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.abg_model = nn.Sequential(
            nn.Linear(64, abg_dim),
            nn.Tanh(),
        )
        # self.tau_model = nn.Sequential(nn.Linear(64, tau_dim))
        self.v_model = nn.Sequential(nn.Linear(64, 1))

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        feature = self.feature_model(obs)
        abg = self.abg_model(feature)
        # tau = self.tau_model(feature)
        v = self.v_model(feature).reshape(-1)
        return abg, v


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_info_sets,
        device: th.device,
        gamma: float,
        gae_lambda: float
    ):
        self.buffer_size = buffer_size
        self.num_info_sets=num_info_sets
        # self.abg_dim = action_space["abg"].shape[0]
        # self.tau_dim = action_space["tau"].n
        self.abg_dim =3
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # self.obs_shape = observation_space.shape
        self.obs_dim=4
        # self.n_envs = n_envs
        # self.data_reshaped = False
        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self.full = False
        #第二个维度改成信息集个数，从此每个RolloutBuffer只储存一个游戏
        self.obs = np.zeros(
            (self.buffer_size, self.num_info_sets, self.obs_dim), dtype=np.float64
        )
        self.acts_abg = np.zeros(
            (self.buffer_size, self.num_info_sets, self.abg_dim), dtype=np.float64
        )
        # self.acts_tau = np.zeros((self.buffer_size, self.num_info_sets), dtype=np.float64)
        self.rews = np.zeros((self.buffer_size, self.num_info_sets), dtype=np.float64)
        self.dones = np.zeros((self.buffer_size, self.num_info_sets), dtype=np.float64)
        self.values = np.zeros((self.buffer_size, self.num_info_sets), dtype=np.float64)
        self.log_probs = np.zeros((self.buffer_size, self.num_info_sets), dtype=np.float64)
        self.advs = np.zeros((self.buffer_size, self.num_info_sets), dtype=np.float64)
        self.rets = np.zeros((self.buffer_size, self.num_info_sets), dtype=np.float64)
        # self.data_reshaped = False

    def add(
        self,
        obs: np.ndarray,
        act_abgs: np.ndarray,
        rew: np.ndarray,
        done: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        self.obs[self.pos] = obs
        self.acts_abg[self.pos] = act_abgs
        # self.acts_tau[self.pos] = act_tau
        self.rews[self.pos] = rew
        self.dones[self.pos] = done
        self.values[self.pos] = values
        self.log_probs[self.pos] = log_probs

        self.pos += 1
        if self.pos == self.buffer_size:
            self.pos = 0
            self.full = True

    def compute_returns_and_advantage(self, eventual_value):
        last_value = eventual_value
        last_gae = 0
        for step in reversed(range(self.buffer_size)):
            delta = (
                self.rews[step]
                + self.gamma * last_value * (1 - self.dones[step])
                - self.values[step]
            )
            last_gae = (
                delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * last_gae
            )
            self.advs[step] = last_gae
            self.rets[step] = self.advs[step] + self.values[step]
            last_value = self.values[step]

    # def get(
    #     self,
    #     batch_size: Optional[int] = None,
    # ) -> "RolloutBufferSamples":
    #     if batch_size is None:
    #         batch_size = self.buffer_size * self.n_envs
    #     indices = np.random.permutation(self.buffer_size * self.n_envs)
    #     start_idx = 0
    #
    #     if not self.data_reshaped:
    #         keys = [
    #             "obs",
    #             "acts_abg",
    #             "acts_tau",
    #             "values",
    #             "log_probs",
    #             "advs",
    #             "rets",
    #         ]
    #         for key in keys:
    #             self.__dict__[key] = self.swap_and_flatten(self.__dict__[key])
    #         self.data_reshaped = True
    #
    #     while start_idx < self.buffer_size * self.n_envs:
    #         data = self._get_samples(indices[start_idx : start_idx + batch_size])
    #         yield data
    #         start_idx += batch_size

    # def _get_samples(self, batch_ids: np.ndarray):
    #     data = (
    #         self.obs[batch_ids],
    #         self.acts_abg[batch_ids].squeeze(),
    #         self.acts_tau[batch_ids].squeeze(),
    #         self.values[batch_ids].squeeze(),
    #         self.log_probs[batch_ids].squeeze(),
    #         self.advs[batch_ids].squeeze(),
    #         self.rets[batch_ids].squeeze(),
    #     )
    #     samples = RolloutBufferSamples(*tuple(map(self.obs_to_tensor, data)))
    #     return samples

    # def obs_to_tensor(self, obs: np.ndarray) -> th.Tensor:
    #     return th.as_tensor(obs).to(self.device)
    #
    # def swap_and_flatten(self, arr: np.ndarray) -> np.ndarray:
    #     """
    #     Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
    #     to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
    #     to [n_steps * n_envs, ...] (which maintain the order)
    #
    #     :param arr:
    #     :return:
    #     """
    #     shape = arr.shape
    #     if len(shape) < 3:
    #         shape = shape + (1,)
    #     return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class RolloutBufferSamples(NamedTuple):
    obs: th.Tensor
    acts_abg: th.Tensor
    # acts_tau: th.Tensor
    old_values: th.Tensor
    old_log_probs: th.Tensor
    advs: th.Tensor
    rets: th.Tensor





class RolloutBuffers:

    def __init__(self,device,buffer_size,n_envs):
        self.reset()
        self.device = device
        self.buffer_size=buffer_size
        self.n_envs=n_envs

    def _swap_and_flatten(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:]).squeeze()

    def add(self, rollout_buffer: "RolloutBuffer") -> None:
        #从 rollout_buffer 中提取数据，扁平化后追加到内部数组。
        keys = [
            "obs", "acts_abg", "rews", "dones",
            "values", "log_probs", "advs", "rets"
        ]

        for key in keys:
            # 从 rollout_buffer 获取数组
            source_array = getattr(rollout_buffer, key)
            # 扁平化数组
            flattened_array = self._swap_and_flatten(source_array)

            # 获取当最终的数组
            dest_array = getattr(self, key)
            # 将扁平化后的新数据拼接到目标数组的末尾
            updated_array = np.concatenate([dest_array, flattened_array], axis=0)
            # 更新当前对象的数组
            setattr(self, key, updated_array)

    def reset(self) -> None:
        self.pos = 0
        self.full = False
        # 集合了所有游戏的数组
        self.obs = np.zeros((0, 4), dtype=np.float64)
        self.acts_abg = np.zeros((0, 3), dtype=np.float64)
        self.rews = np.zeros((0,), dtype=np.float64)
        self.dones = np.zeros((0,), dtype=np.float64)
        self.values = np.zeros((0,), dtype=np.float64)
        self.log_probs = np.zeros((0,), dtype=np.float64)
        self.advs = np.zeros((0,), dtype=np.float64)
        self.rets = np.zeros((0,), dtype=np.float64)
        # self.data_reshaped = False

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> "RolloutBufferSamples":
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        # indices = np.random.permutation(self.buffer_size * self.n_envs)
        total_length=self.obs.shape[0]
        indices = np.random.permutation(total_length)
        start_idx = 0

        # if not self.data_reshaped:
        #     keys = [
        #         "obs",
        #         "acts_abg",
        #         "acts_tau",
        #         "values",
        #         "log_probs",
        #         "advs",
        #         "rets",
        #     ]
        #     for key in keys:
        #         self.__dict__[key] = self.swap_and_flatten(self.__dict__[key])
        #     self.data_reshaped = True

        while start_idx < total_length:
            data = self._get_samples(indices[start_idx : start_idx + batch_size])
            yield data
            start_idx += batch_size

    def _get_samples(self, batch_ids: np.ndarray):
        data = (
            self.obs[batch_ids],
            self.acts_abg[batch_ids].squeeze(),
            # self.acts_tau[batch_ids].squeeze(),
            self.values[batch_ids].squeeze(),
            self.log_probs[batch_ids].squeeze(),
            self.advs[batch_ids].squeeze(),
            self.rets[batch_ids].squeeze(),
        )
        # print(f"原论文obs的维度{self.obs.shape}/n")
        # print(f"values的维度{values.shape}/n")
        # print(f"原论文advs的维度{self.advs.shape}/n")
        samples = RolloutBufferSamples(*tuple(map(self.obs_to_tensor, data)))
        return samples

    def obs_to_tensor(self, obs: np.ndarray) -> th.Tensor:
        return th.tensor(obs, dtype=th.double).to(self.device)

    def swap_and_flatten(self, arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])










