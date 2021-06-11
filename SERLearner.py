import numpy as np

import QLearner
from scipy.optimize import minimize


class SERLearner(QLearner):

    def __init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, obj_fn, opt=False,
                 multi_ce=False, ce_ser=None, single_ce=False, rand_prob=False, ce_sgn=None):
        self.ce_ser = ce_ser
        super().__init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, obj_fn, opt, multi_ce,
                         single_ce, rand_prob, ce_sgn)

    #  calculates SE Returns using self.obj_fn value and provided vector.
    def calc_returns(self, vector):
        ser = 0
        if self.obj_fn == 0:
            ser = vector[0] ** 2 + vector[1] ** 2
        elif self.obj_fn == 1:
            ser = vector[0] * vector[1]
        return ser

    def calc_mixed_strategy_nonlinear(self, state):
        if self.rand_prob:
            s0 = np.random.random(self.num_actions)
            s0 /= np.sum(s0)
        else:
            s0 = np.full(self.num_actions,
                         1.0 / self.num_actions)  # initial guess set to equal prob over all actions

        b = (0.0, 1.0)
        bnds = (b,) * self.num_actions
        con1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        cons = ([con1])
        solution = minimize(self.objective, s0, bounds=bnds, constraints=cons)
        strategy = solution.x

        if self.single_CE:
            if strategy[state] > 0:
                return state
            else:
                return strategy

        # if this solution has the same SER as the CE_strategy, choose the CE one.
        if self.multi_CE:
            max_ser = self.calc_ser_from_strategy(strategy)
            if max_ser < self.ce_ser:
                return state
        return strategy

    def calc_ser_from_strategy(self, strategy):
        expected_vec = self.calc_expected_vec(self.current_state, strategy)
        ser = self.calc_returns(self.agent_id, expected_vec)
        return ser

    def objective(self, strategy):
        return - self.calc_ser_from_strategy(strategy)


def calc_ser(agent, vector):
    ser = 0
    if agent == 0:
        ser = vector[0] ** 2 + vector[1] ** 2
    elif agent == 1:
        ser = vector[0] * vector[1]
    return ser
