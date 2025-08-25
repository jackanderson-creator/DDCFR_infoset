from typing import List
import numpy as np
from ddcfr.cfr.cfr import CFRSolver, CFRState
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger

class DCFRState(CFRState):
    def __init__(self, legal_actions: List[int], current_player: int, alpha: float = 1.5, beta: float = 0, gamma: float = 2):
        super().__init__(legal_actions, current_player)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.vanilla_cfr_regrets = {a: 0 for a in self.legal_actions}

    def cumulate_vanilla_cfr_regret(self):
        for a in self.regrets.keys(): self.vanilla_cfr_regrets[a] += self.imm_regrets[a]

    def cumulate_regret(self, T, alpha, beta):
        T = float(T)
        for a in self.regrets.keys():
            if T == 1: self.regrets[a] = self.imm_regrets[a]; continue
            if self.regrets[a] > 0: self.regrets[a] = self.regrets[a] * (np.power(T - 1, alpha) / (np.power(T - 1, alpha) + 1)) + self.imm_regrets[a]
            else: self.regrets[a] = self.regrets[a] * (np.power(T - 1, beta) / (np.power(T - 1, beta) + 1)) + self.imm_regrets[a]

    def cumulate_policy(self, T, gamma):
        T = float(T)
        CLIP_THRESHOLD = 1e20
        for a in self.regrets.keys():
            if T == 1: self.cum_policy[a] = self.reach * self.policy[a]; continue
            updated_policy = self.cum_policy[a] * np.power((T - 1) / T, gamma) + self.reach * self.policy[a]
            self.cum_policy[a] = np.clip(updated_policy, -CLIP_THRESHOLD, CLIP_THRESHOLD)

class DCFRSolver(CFRSolver):
    def __init__(self, game_config: GameConfig, logger: Logger, alpha: float = 1.5, beta: float = 0, gamma: float = 2):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        super().__init__(game_config, logger)

    def _init_state(self, h):
        return DCFRState(h.legal_actions(), h.current_player(), self.alpha, self.beta, self.gamma)

    def update_state(self, s: DCFRState):
        s.cumulate_regret(self.num_iterations, s.alpha, s.beta)
        s.cumulate_vanilla_cfr_regret()
        s.cumulate_policy(self.num_iterations, s.gamma)
        s.update_current_policy()

class DDCFRSolver(DCFRSolver):
    def __init__(self, game_config: GameConfig, logger: Logger):
        super().__init__(game_config, logger)
        self.ordered_states = list(self.states.values())
        self.state_to_idx = {state: i for i, state in enumerate(self.ordered_states)}
        self.total_iterations = game_config.iterations
        self.alpha_range, self.beta_range, self.gamma_range = [0, 5], [-5, 0], [0, 5]
        self.last_log_conv, self.start_log_conv = None, None
        self.previous_vanilla_regrets = {}
        # self.local_reward_weight = local_reward_weight

    def _cache_vanilla_regrets(self):
        self.previous_vanilla_regrets = {
            i: s.vanilla_cfr_regrets.copy() for i, s in enumerate(self.ordered_states)
        }

    def reset_for_new_run(self):
        self.states = {}
        self._init_states(self.game.new_initial_state())
        self.ordered_states = list(self.states.values())
        self.state_to_idx = {state: i for i, state in enumerate(self.ordered_states)}
        self.conv_history = []
        self.num_nodes_touched = 0
        self.num_iterations = 0

        initial_conv = self.calc_conv()
        self.conv_history.append(initial_conv)
        log_conv = np.log(initial_conv) / np.log(10)
        self.start_log_conv = log_conv
        self.last_log_conv = log_conv
        
        # 初始化“上一步”的遗憾值为全零
        self.previous_vanilla_regrets = {
            i: {a: 0.0 for a in s.legal_actions} for i, s in enumerate(self.ordered_states)
        }

    def iteration(self, alphas, betas, gammas):
        for i in range(self.np):
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)
            pending_states = [s for s in self.states.values() if s.player == i]
            for s in pending_states:
                k = self.state_to_idx.get(s)
                if k is not None:
                    s.alpha, s.beta, s.gamma = alphas[k], betas[k], gammas[k]
                    self.update_state(s)
                s.clear_temp()

    def get_all_infoset_states_for_ppo(self):
        if self.start_log_conv is None or self.last_log_conv is None: self.reset_for_new_run()
        iters_norm = self.num_iterations / self.total_iterations
        conv_range = self.start_log_conv - (-12)
        conv_frac = (self.last_log_conv - (-12)) / (conv_range + 1e-12)
        states_features = []
        for s in self.ordered_states:
            # regrets = list(s.regrets.values())
            #对遗憾进行尺度放缩！！！！！！！！！！！！！！！！！！！
            # if regrets:
            #     max_regret, min_regret = (np.max(regrets), np.min(regrets)) 
            #     if max_regret!=0:
            #         max_regret=np.sign(max_regret)*(np.log(np.abs(max_regret)*1e20)/np.log(10))
            #     if min_regret!=0:
            #         min_regret=np.sign(min_regret)*(np.log(np.abs(min_regret)*1e20)/np.log(10))
            #     max_regret=max_regret/10.0
            #     min_regret=min_regret/10.0
            # else:
            #     max_regret, min_regret = (0.0, 0.0)

            states_features.append((iters_norm, conv_frac))
        return np.nan_to_num(np.array(states_features, dtype=np.float64))

    def run_one_cfr_iteration(self, alphas, betas, gammas):
        self.num_iterations += 1
        if self.num_iterations > self.total_iterations: return
        self.iteration(alphas, betas, gammas)

    def run_one_ppo_step(self, actions_from_ppo):
        unscaled_actions = self._unscale_actions(actions_from_ppo)
        alphas, betas, gammas = unscaled_actions.T
        iterations_per_step = 2
        for _ in range(iterations_per_step):
            if self.num_iterations >= self.total_iterations: break
            self.run_one_cfr_iteration(alphas, betas, gammas)
        
        self.after_iteration("iterations", eval_iterations_interval=1, dump_log=False)
        
        new_log_conv = np.log(self.conv_history[-1]) / np.log(10)
        # global_reward = self.last_log_conv - new_log_conv
        

        #对奖励进行归一化！！！！！！！！！！！！！！！！！！！！！
        conv_range = self.start_log_conv - (-12)
        global_reward = (self.last_log_conv - new_log_conv) / (conv_range + 1e-12)

        self.last_log_conv = new_log_conv
        
        # ==================== 代码修改部分 START ====================
        # 根据要求，移除了局部惩罚项的计算逻辑。
        # 现在，我们为每一个信息集都提供完全相同的全局奖励。
        
        # 获取信息集的总数
        num_infosets = len(self.ordered_states)
        # 创建一个数组，其所有元素的值都是当前的 global_reward
        rewards = np.full(num_infosets, global_reward, dtype=np.float64)

        # 由于不再计算局部惩罚，所以也无需为下一步缓存遗憾值。
        # 因此，`_cache_vanilla_regrets()` 方法的调用已被移除。
        
        done = self.num_iterations >= self.total_iterations
        return rewards, done
        # ==================== 代码修改部分 END ======================

    def _unscale_actions(self, actions):
        unscaled = np.zeros_like(actions)
        unscaled[:, 0] = self._denormalize(actions[:, 0], *self.alpha_range)
        unscaled[:, 1] = self._denormalize(actions[:, 1], *self.beta_range)
        unscaled[:, 2] = self._denormalize(actions[:, 2], *self.gamma_range)
        return unscaled

    def _denormalize(self, param, param_min, param_max):
        return 0.5 * (param + 1) * (param_max - param_min) + param_min