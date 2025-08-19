import random
from typing import List, Optional, Union
from typing import Tuple

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv

from ddcfr.cfr import DDCFRSolver
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger
from ddcfr.utils.utils import load_module


class CFREnv(gym.Env):
    def __init__(
        self,
        game_config: GameConfig,
        logger: Logger,
        total_iterations: int,
        seed: int = 0,
        reward_lambda: float = 1.0,  # 新增: λ
        target_iters_range: Tuple[float, float] = (1e3, 1e4),  # 新增: N range
        target_conv_range: Tuple[float, float] = (0.00001, 0.1),  # 新增: ε range
    ):
        super().__init__()
        self.game_config = game_config
        self.logger = logger
        self.total_iterations = total_iterations
        self.game_name = game_config.name
        self.eval_iterations_interval = 1
        self.action_space = spaces.Dict(
            {
                "abg": spaces.Box(low=-1, high=1, shape=[3], dtype=np.float64),
                "tau": spaces.Discrete(5),
            }
        )
        self.tau_list = [1, 2, 5, 10, 20]
        self.alpha_range = [0, 5]
        self.beta_range = [-5, 0]
        self.gamma_range = [0, 5]
        self.observation_space = spaces.Box(low=0, high=1, shape=[4], dtype=np.float64)  # 修改: 4维
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.reward_lambda = reward_lambda  # 新增
        self.target_iters_range = target_iters_range  # 新增
        self.target_conv_range = target_conv_range  # 新增

    def reset(self):
        self.solver = DDCFRSolver(self.game_config, self.logger)
        self.solver.after_iteration(
            "iterations",
            eval_iterations_interval=self.eval_iterations_interval,
        )
        conv = self.solver.conv_history[-1]
        log_conv = np.log(conv) / np.log(10)
        self.start_log_conv = log_conv
        self.last_log_conv = log_conv
        # 新增: 采样goal
        self.target_iters = np.random.uniform(*self.target_iters_range)
        self.target_conv = np.random.uniform(*self.target_conv_range)
        iters_norm = self._normalize_iters(self.solver.num_iterations)
        conv_frac = self.calc_conv_frac(log_conv)
        target_iters_norm = self._normalize_iters(self.target_iters)
        target_conv_frac = self.calc_conv_frac(np.log(self.target_conv) / np.log(10))
        state = (iters_norm, conv_frac, target_iters_norm, target_conv_frac)
        return np.array(state, dtype=np.float64)

    def step(self, action):
        alpha, beta, gamma, tau = self._unscale_action(action)
        self.logger.record(f"{self.game_name}/tau", tau)
        for _ in range(tau):
            self.solver.num_iterations += 1
            self.solver.iteration(alpha, beta, gamma)
            self.logger.record(f"{self.game_name}/alpha", alpha)
            self.logger.record(f"{self.game_name}/beta", beta)
            self.logger.record(f"{self.game_name}/gamma", gamma)
            self.solver.after_iteration(
                "iterations",
                eval_iterations_interval=self.eval_iterations_interval,
            )
            if self.solver.num_iterations == self.total_iterations:
                break
        conv = self.solver.conv_history[-1]
        log_conv = np.log(conv) / np.log(10)
        # 修改: done条件扩展
        done = (self.solver.num_iterations >= self.total_iterations) or (conv <= self.target_conv)
        # 修改: 奖励计算
        base_reward = self.last_log_conv - log_conv
        speed_bonus = self.reward_lambda * (1 - self.solver.num_iterations / self.target_iters)
        reward = base_reward + speed_bonus
        if self.solver.num_iterations > self.target_iters and conv > self.target_conv:
            reward -= 10  # 大罚款
        self.last_log_conv = log_conv
        iters_norm = self._normalize_iters(self.solver.num_iterations)
        conv_frac = self.calc_conv_frac(log_conv)
        target_iters_norm = self._normalize_iters(self.target_iters)
        target_conv_frac = self.calc_conv_frac(np.log(self.target_conv) / np.log(10))
        state = (iters_norm, conv_frac, target_iters_norm, target_conv_frac)
        info = {"start_log_conv": self.start_log_conv}
        return np.array(state, dtype=np.float64), reward, done, info

    def _unscale_action(self, action):
        alpha, beta, gamma = action["abg"]
        tau = action["tau"]
        alpha = self.denormalize(alpha, *self.alpha_range)
        beta = self.denormalize(beta, *self.beta_range)
        gamma = self.denormalize(gamma, *self.gamma_range)
        tau = self.tau_list[tau]
        return alpha, beta, gamma, tau

    def denormalize(self, param, param_min, param_max):
        param_mid = (param_max + param_min) / 2
        param_half_len = (param_max - param_min) / 2
        param = param * param_half_len + param_mid
        return param

    def calc_conv_frac(self, log_conv):
        start_log_conv = self.start_log_conv
        final_log_conv = -12
        conv_frac = (log_conv - final_log_conv) / (start_log_conv - final_log_conv)
        return conv_frac

    def _normalize_iters(self, iters):
        return iters / self.total_iterations  # 对于target_iters，也用total_iterations归一化（假设total_iterations是上界）


class MultiCFREnv(CFREnv):
    def __init__(
        self,
        game_configs: List[GameConfig],
        logger: Logger,
        seed: int = 0,
        reward_lambda: float = 1.0,  # 新增
        target_iters_range: Tuple[float, float] = (1e3, 1e4),  # 新增
        target_conv_range: Tuple[float, float] = (0.00001, 0.1),  # 新增
    ):
        super().__init__(game_configs[0], logger, game_configs[0].iterations, seed,
                         reward_lambda, target_iters_range, target_conv_range)
        self.game_configs = game_configs

    def reset(self):
        self.game_config = random.choice(self.game_configs)
        self.total_iterations = self.game_config.iterations
        self.game_name = self.game_config.name
        return super().reset()


class ConvStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        self.episode_returns = 0
        return state

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.episode_returns += reward
        info["returns"] = self.episode_returns
        info["conv"] = np.exp(
            (-self.episode_returns + info["start_log_conv"]) * np.log(10)
        )
        real_conv = self.env.solver.conv_history[-1]
        assert abs(info["conv"] - real_conv) < 1e-8
        # diff = abs(info["conv"] - real_conv)
        # if diff >= 1e-8:
        #     print(f"666出错了[ERROR] conv mismatch: info={info['conv']}, real={real_conv}, diff={diff}")

        if done:
            self.episode_returns = 0
        return state, reward, done, info


def make_cfr_env(
    game_name: Union[str, GameConfig],
    total_iterations: int = 10,
    writer_strings: List[str] = [],
    logger: Optional[Logger] = None,
    reward_lambda: float = 1.0,  # 新增
    target_iters_range: Tuple[float, float] = (1e3, 1e3+101),  # 新增
    target_conv_range: Tuple[float, float] = (0.00001, 0.1),  # 新增
):
    if isinstance(game_name, str):
        game_class = load_module(f"ddcfr.game.game_config:{game_name}")
        game_config = game_class()
    elif isinstance(game_name, GameConfig):
        game_config = game_name
    else:
        raise ValueError("game name should be a string or a GameConfig")
    if logger is None:
        logger = Logger(writer_strings)
    env = CFREnv(game_config, logger, total_iterations,
                 reward_lambda=reward_lambda,
                 target_iters_range=target_iters_range,
                 target_conv_range=target_conv_range)
    env = ConvStatistics(env)
    return env


def make_multi_cfr_env(game_configs: List[GameConfig],
                       reward_lambda: float = 1.0,  # 新增
                       target_iters_range: Tuple[float, float] = (1e3, 1e4),  # 新增
                       target_conv_range: Tuple[float, float] = (0.00001, 0.1),  # 新增
):
    logger = Logger([])
    env = MultiCFREnv(game_configs, logger,
                      reward_lambda=reward_lambda,
                      target_iters_range=target_iters_range,
                      target_conv_range=target_conv_range)
    env = ConvStatistics(env)
    return env


def make_cfr_vec_env(
    game_configs: List[GameConfig],
    n_envs: int = 2,
    reward_lambda: float = 1.0,  # 新增
    target_iters_range: Tuple[float, float] = (1e3, 1e4),  # 新增
    target_conv_range: Tuple[float, float] = (0.00001, 0.1),  # 新增
):
    env = SubprocVecEnv(
        [lambda: make_multi_cfr_env(game_configs,
                                    reward_lambda=reward_lambda,
                                    target_iters_range=target_iters_range,
                                    target_conv_range=target_conv_range) for i in range(n_envs)]
    )
    return env