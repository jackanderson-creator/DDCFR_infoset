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
        self.ordered_states = list(self.states.values())
        self.state_to_idx = {state: i for i, state in enumerate(self.ordered_states)}

    def _init_state(self, h):
        return DCFRState(
            h.legal_actions(),
            h.current_player(),
        )

    #alpha现在不是一个数，而是列表
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

    # def iteration(self, alpha: float = 1.5, beta: float = 0, gamma: float = 2):
    #     for i in range(self.np):
    #         h = self.game.new_initial_state()
    #         self.calc_regret(h, i, 1, 1)
    #
    #         pending_states = [s for s in self.states.values() if s.player == i]
    #
    #         for s in pending_states:
    #             s.alpha = alpha
    #             s.beta = beta
    #             s.gamma = gamma
    #             self.update_state(s)
    #             s.clear_temp()
