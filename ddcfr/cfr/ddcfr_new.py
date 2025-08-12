from ddcfr.cfr.dcfr import DCFRSolver, DCFRState
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger


class DDCFRSolver(DCFRSolver):
    def __init__(
        self,
        game_config: GameConfig,
        logger: Logger,
    ):
        super().__init__(game_config, logger)

    def _init_state(self, h):
        return DCFRState(
            h.legal_actions(),
            h.current_player(),
        )

    def iteration(self,alphas, betas, gammas):
        k=0
        for i in range(self.np):
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)

            pending_states = [s for s in self.states.values() if s.player == i]

            for s in pending_states:
                assert s==self.states.values()[k],"不对应！！"
                # ave_regret = np.mean(s.regrets)
                alpha=alphas[k]
                beta = betas[k]
                gamma= gammas[k]
                # alpha,beta,gamma,_=ppo.policy(ave_regret)
                s.alpha = alpha
                s.beta = beta
                s.gamma = gamma
                self.update_state(s)
                s.clear_temp()
                k+=1



    def iteration_start(self,alpha: float = 1.5, beta: float = 0, gamma: float = 2):
        for i in range(self.np):
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)

            pending_states = [s for s in self.states.values() if s.player == i]

            for s in pending_states:
                s.alpha = alpha
                s.beta = beta
                s.gamma = gamma
                self.update_state(s)
                s.clear_temp()
