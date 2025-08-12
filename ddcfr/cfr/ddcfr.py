from ddcfr.cfr.dcfr import DCFRSolver, DCFRState
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger
import numpy as np

class DDCFRSolver(DCFRSolver):
    def __init__(
        self,
        game_config: GameConfig,
        logger: Logger,
    ):
        super().__init__(game_config, logger)
        # 在初始化时创建一次性的、有序的状态列表和映射
        self.ordered_states = list(self.states.values())
        self.state_to_idx = {state: i for i, state in enumerate(self.ordered_states)}
        self.total_iterations = game_config.iterations
        self.alpha_range = [0, 5]
        self.beta_range = [-5, 0]
        self.gamma_range = [0, 5]
        self.last_log_conv = None
        self.start_log_conv = None

    def _init_state(self, h):
        return DCFRState(h.legal_actions(), h.current_player())

    def reset_for_new_run(self):
        """重置solver以开始一个全新的训练回合"""
        self.states = {}
        self._init_states(self.game.new_initial_state())
        self.ordered_states = list(self.states.values())
        self.state_to_idx = {state: i for i, state in enumerate(self.ordered_states)}
        self.conv_history = []
        self.num_nodes_touched = 0
        self.num_iterations = 0

        # 仅计算并设置初始的exploitability值
        initial_conv = self.calc_conv()
        self.conv_history.append(initial_conv)

        log_conv = np.log(initial_conv) / np.log(10)
        self.start_log_conv = log_conv
        self.last_log_conv = log_conv

    def iteration(self, alphas, betas, gammas):
        """这是核心的CFR迭代，现在由run_one_ppo_step调用"""
        for i in range(self.np):
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)

            pending_states = [s for s in self.states.values() if s.player == i]

            for s in pending_states:
                k = self.state_to_idx[s]
                
                s.alpha = alphas[k]
                s.beta = betas[k]
                s.gamma = gammas[k]

                self.update_state(s)
                s.clear_temp()

    def get_all_infoset_states_for_ppo(self):
        """为PPO收集所有信息集的状态"""
        if self.start_log_conv is None or self.last_log_conv is None:
            self.reset_for_new_run()

        iters_norm = self.num_iterations / self.total_iterations
        # 防止除以零
        conv_range = self.start_log_conv - (-12)
        conv_frac = (self.last_log_conv - (-12)) / (conv_range + 1e-8)

        states_features = [
            (iters_norm, conv_frac, np.mean(list(s.regrets.values())))
            for s in self.ordered_states
        ]
        return np.nan_to_num(np.array(states_features, dtype=np.float64))

    def run_one_ppo_step(self, actions_from_ppo):
        """执行一个PPO步骤，包含多次CFR迭代"""
        unscaled_actions = self._unscale_actions(actions_from_ppo)
        alphas, betas, gammas = unscaled_actions.T
        
        iterations_per_step = 20
        for _ in range(iterations_per_step):
            self.num_iterations += 1
            if self.num_iterations > self.total_iterations:
                break
            self.iteration(alphas, betas, gammas)
        
        # 在这里只计算和记录数据，但不dump日志
        self.after_iteration("iterations", eval_iterations_interval=1, dump_log=False)
        
        new_log_conv = np.log(self.conv_history[-1]) / np.log(10)
        reward = self.last_log_conv - new_log_conv
        self.last_log_conv = new_log_conv
        
        done = self.num_iterations >= self.total_iterations
        
        return reward, done

    def _unscale_actions(self, actions):
        """将PPO输出的tanh动作 (-1, 1) 转换到实际范围"""
        unscaled = np.zeros_like(actions)
        unscaled[:, 0] = self._denormalize(actions[:, 0], *self.alpha_range)
        unscaled[:, 1] = self._denormalize(actions[:, 1], *self.beta_range)
        unscaled[:, 2] = self._denormalize(actions[:, 2], *self.gamma_range)
        return unscaled

    def _denormalize(self, param, param_min, param_max):
        """将-1到1范围的值映射到(min, max)"""
        return 0.5 * (param + 1) * (param_max - param_min) + param_min