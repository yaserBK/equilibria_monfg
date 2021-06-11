import QLearner

class ESRLeaner(QLearner):

    def __init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt=False,
                 multi_ce=False, ce_esr=None, single_ce=False, rand_prob=False, ce_sgn=None):
        self.ce_esr = ce_esr
        super().__init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt=opt,
                         multi_ce=multi_ce, single_ce=single_ce, rand_prob=rand_prob, ce_sgn=ce_sgn)

    def calc_esr(agent, vector):
        pass

    def calc_mixed_strategy_nonlinear(self, state):
        pass

    def calc_esr_from_strategy(self, strategy):
        pass

    def objective(self, strategy):
        pass