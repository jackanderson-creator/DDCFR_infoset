import random
from typing import List, Optional, Union

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
    ):
        super().__init__()
        self.game_config = game_config
        self.logger = logger
        self.total_iterations = total_iterations
        self.game_name = game_config.name
        self.eval_iterations_interval = 1

        # 初始化solver，获取信息集的数量
        self.solver = DDCFRSolver(self.game_config, self.logger)
        self.info_set_keys = list(self.solver.states.keys())
        self.num_info_sets = len(self.info_set_keys)

        # self.action_space = spaces.Dict(
        #     {
        #         "abg": spaces.Box(low=-1, high=1, shape=[3], dtype=np.float64),
        #         "tau": spaces.Discrete(5),
        #     }
        # )

        # self.action_space=spaces.Box(low=-1, high=1, shape=(self.num_info_sets,3), dtype=np.float64)

        self.action_space = spaces.Dict(
            {
                "abg": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(self.num_info_sets, 3),
                    dtype=np.float64
                ),
                "tau": spaces.MultiDiscrete([5] * self.num_info_sets),
            }
        )

        self.tau_list = [1, 2, 5, 10, 20]
        self.alpha_range = [0, 5]
        self.beta_range = [-5, 0]
        self.gamma_range = [0, 5]
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_info_sets,4), dtype=np.float64)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_info_sets, 2), dtype=np.float64)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

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
        iters = self._normalize_iters(self.solver.num_iterations)
        conv_frac = self.calc_conv_frac(log_conv)
        state=[]
        # for s in list(self.solver.states.values()):
        for s in self.solver.ordered_states:
            # min_regret=min(s.regrets)
            # max_regret = max(s.regrets)
            state.append([iters, conv_frac])
        # state = (iters, conv_frac)
        return np.array(state, dtype=np.float64)

    def step(self, action):
        # action中没有了tau
        alpha, beta, gamma,tau= self._unscale_action(action)
        #tau固定
        # tau=2
        self.logger.record(f"{self.game_name}/tau", tau)
        for _ in range(tau):
            self.solver.num_iterations += 1
            self.solver.iteration(alpha, beta, gamma)
            #暂时只记录第一个信息集上的动作
            self.logger.record(f"{self.game_name}/alpha_infoset0", alpha[0])
            self.logger.record(f"{self.game_name}/beta_infoset0", beta[0])
            self.logger.record(f"{self.game_name}/gamma_infoset0", gamma[0])
            self.solver.after_iteration(
                "iterations",
                eval_iterations_interval=self.eval_iterations_interval,
            )
            if self.solver.num_iterations == self.total_iterations:
                break
        conv = self.solver.conv_history[-1]
        log_conv = np.log(conv) / np.log(10)
        done = self.solver.num_iterations == self.total_iterations
        reward = self.last_log_conv - log_conv
        self.last_log_conv = log_conv
        iters = self._normalize_iters(self.solver.num_iterations)
        conv_frac = self.calc_conv_frac(log_conv)
        #状态修改为多个信息集的
        state = []
        # for s in list(self.solver.states.values()):
        for s in self.solver.ordered_states:
            # min_regret=min(s.regrets)
            # max_regret = max(s.regrets)
            state.append([iters, conv_frac])

        info = {"start_log_conv": self.start_log_conv}
        return np.array(state, dtype=np.float64), reward, done, info

    def _unscale_action(self, action):
        abg=action["abg"]
        alpha, beta, gamma = abg.T
        alpha = self.denormalize(alpha, *self.alpha_range)
        beta = self.denormalize(beta, *self.beta_range)
        gamma = self.denormalize(gamma, *self.gamma_range)
        #选取第一个信息集的tau
        tau=action["tau"][0]
        tau = self.tau_list[tau]
        return alpha, beta, gamma,tau

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
        return iters / self.total_iterations


class MultiCFREnv(CFREnv):
    def __init__(
        self,
        game_configs: List[GameConfig],
        logger: Logger,
        i_env,
        seed: int = 0,
    ):
        self.game_configs = game_configs
        # self.game_config = random.choice(self.game_configs)
        #不随机选了
        self.game_config = self.game_configs[i_env%4]
        self.total_iterations = self.game_config.iterations
        self.game_name = self.game_config.name
        super().__init__(self.game_config, logger, self.game_config.iterations, seed)

    def reset(self):
        # self.game_config = random.choice(self.game_configs)
        # self.total_iterations = self.game_config.iterations
        # self.game_name = self.game_config.name
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
        if done:
            # self.episode_returns = 0
            info["terminal_observation"]=state
            state=self.reset()
            
        return state, reward, done, info


def make_multi_cfr_env(game_configs: List[GameConfig],i_env):
    logger = Logger([])
    env = MultiCFREnv(game_configs, logger,i_env)
    env = ConvStatistics(env)
    return env


def make_cfr_vec_env(
    game_configs: List[GameConfig],
    n_envs: int = 2,
):
    env = SubprocVecEnv(
        [lambda: make_multi_cfr_env(game_configs) for i in range(n_envs)]
    )
    return env


def make_cfr_env(
    game_name: Union[str, GameConfig],
    total_iterations: int = 10,
    writer_strings: List[str] = [],
    logger: Optional[Logger] = None,
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
    env = CFREnv(game_config, logger, total_iterations)
    env = ConvStatistics(env)
    return env
