import numpy as np

from QLearner import QLearner


class ESRLearner(QLearner):

    def __init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, obj_fn, obj1Payoffs, obj2Payoffs, opt=False,
                 multi_ce=False, ce_esr=None, single_ce=False, rand_prob=False, ce_sgn=None, ):
        self.ce_esr = ce_esr
        self.obj1Payoffs =obj1Payoffs
        self.obj2Payoffs = obj2Payoffs

        # Scalarizing payoff tables
        if agent_id == 0:
            self.sclarized_payoffs = obj1Payoffs*obj2Payoffs
        elif agent_id == 1:
            self.scalarized_payoffs = np.square(obj1Payoffs)+np.square(obj2Payoffs)


        super().__init__(agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt, multi_ce,
                         single_ce, rand_prob, ce_sgn)

    def calc_returns(self, agent, vector):
        pass

    def calc_mixed_strategy_nonlinear(self, state):
        pass

    def calc_esr_from_strategy(self, strategy):
        pass

    def objective(self, strategy):
        pass

    def calc_expected_vec(self, state, strategy):
        expected_scalar = None
        if not self.multi_CE:
            expected_scalar =



def calc_esr(agent, vector):
    # TODO: Modify this segment to calculate ESR using agent value and a scalar qty
    # TODO: THE AREA OF CODE WHERE THIS IS CALLED WILL NEED SIGNIFICANT MODIFICATION,  PARTICULARLY FOR INSTANCES WHERE
    # AGENTS WITH THE SAME OBJECTIVE FN ARE CALLED.
    pass
