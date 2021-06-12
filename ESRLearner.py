from QLearner import QLearner


class ESRLearner(QLearner):

    def __init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, obj_fn, opt=False,
                 multi_ce=False, ce_esr=None, single_ce=False, rand_prob=False, ce_sgn=None):
        self.ce_esr = ce_esr
        super().__init__(agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, obj_fn,
                         opt,
                         multi_ce, single_ce, rand_prob, ce_sgn)

    def calc_returns(self, agent, vector):
        pass

    def calc_mixed_strategy_nonlinear(self, state):
        pass

    def calc_esr_from_strategy(self, strategy):
        pass

    def objective(self, strategy):
        pass


def calc_esr(agent, vector):
    # TODO: Modify this segment to calculate ESR using agent value and a scalar qty
    # TODO: THE AREA OF CODE WHERE THIS IS CALLED WILL NEED SIGNIFICANT MODIFICATION,  PARTICULARLY FOR INSTANCES WHERE
    # AGENTS WITH THE SAME OBJECTIVE FN ARE CALLED.
    pass
