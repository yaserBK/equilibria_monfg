import random
import numpy as np
from scipy.optimize import minimize


class QLearnerESR:
    def __init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt=False,
                 multi_ce=False, ce_esr=None, single_ce=False, rand_prob=False, ce_sgn=None):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = 1
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_states, num_actions, self.num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_states, num_actions, self.num_objectives))
        self.current_state = -1
        self.multi_CE = multi_ce
        self.ce_esr = ce_esr
        self.single_CE = single_ce
        self.rand_prob = rand_prob
        self.ce_sgn = ce_sgn
        self.sc = 0

    def update_q_table(self, prev_state, action, curr_state, reward):
        scalarized_reward = self.scalarize(reward)
        old_q = self.q_table[prev_state][action]
        next_q = self.q_table[curr_state, :]
        new_q = np.zeros(self.num_objectives)
        for o in range(self.num_objectives):
            new_q[o] = old_q[o] + self.alpha * (scalarized_reward + self.gamma * max(next_q[:, o]) - old_q[o])
            self.q_table[prev_state][action][o] = new_q[o]

    # random action selection
    def select_random_action(self):
        random_action = np.random.randint(self.num_actions)
        return random_action

    # epsilon greedy based on nonlinear optimiser mixed strategy search
    def select_action_mixed_nonlinear(self, state):
        self.current_state = state
        if random.uniform(0.0, 1.0) < self.epsilon:
            return self.select_random_action()
        else:
            return self.select_action_greedy_mixed_nonlinear(state)

    # greedy action selection based on nonlinear optimiser mixed strategy search
    def select_action_greedy_mixed_nonlinear(self, state):
        strategy = self.calc_mixed_strategy_nonlinear(state)
        if isinstance(strategy, int) or isinstance(strategy, np.int64):
            return strategy
        else:
            if np.sum(strategy) != 1:
                strategy = strategy / np.sum(strategy)
            return np.random.choice(range(self.num_actions), p=strategy)

    def calc_mixed_strategy_nonlinear(self, state):
        if self.rand_prob:
            s0 = np.random.random(self.num_actions)
            s0 /= np.sum(s0)
        else:
            s0 = np.full(self.num_actions, 1.0 / self.num_actions)  # initial guess set to equal prob over all actions

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
            max_esr = self.calc_esr_from_strategy(strategy)
            if max_esr < self.ce_esr:
                return state
        return strategy

    # this is the objective function to be minimised by the nonlinear optimiser
    # (therefore it returns the negative of SER)
    def objective(self, strategy):
        return - self.calc_esr_from_strategy(strategy)

    # Calculates the SER for a given strategy using the agent's own Q values
    def calc_esr_from_strategy(self, strategy):
        esr = self.calc_expected_scalar(self.current_state, strategy)
        return esr

    def calc_expected_scalar(self, state, strategy):
        expected_scal = None
        if not self.multi_CE:
            for o in range(self.num_objectives):
                expected_scal = np.dot(self.q_table[state, :, o], np.array(strategy))
        else:
            expected_tmp = sum(self.ce_sgn[i] * self.q_table[i, :] for i in range(len(self.ce_sgn)))
            # print("Expectation over signals:", expected_tmp)
            for o in range(self.num_objectives):
                expected_scal = np.dot(expected_tmp[:], np.array(strategy))
        return expected_scal

    @staticmethod
    def select_recommended_action(state):
        return state

    # FOR MONFG
    # def scalarize(self, rewards):
    #     scalar = None
    #     if self.agent_id == 0:
    #         scalar = (rewards[0] * rewards[0]) + (rewards[1] * rewards[1])
    #     elif self.agent_id == 1:
    #         scalar = rewards[0] * rewards[1]
    #     return scalar

    # For single objective NFG optimization.
    def scalarize(self, rewards):
        scalar = None
        if self.agent_id == 0:
            scalar = rewards[0]
        elif self.agent_id == 1:
            scalar = rewards[1]
        return scalar


def calc_esr(agent, vector):
    pass


def softmax(q):
    soft_q = np.exp(q - np.max(q))
    return soft_q / soft_q.sum(axis=0)
