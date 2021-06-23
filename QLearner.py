import abc
import random
import numpy as np
from scipy.optimize import minimize
import warnings
# from utils import *
import itertools
from abc import ABC, abstractmethod


class QLearner(ABC):
    def __init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt=False,
                 multi_ce=False, single_ce=False, rand_prob=False, ce_sgn=None):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_states, num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_states, num_actions, num_objectives))
        self.current_state = -1
        self.multi_CE = multi_ce
        #  self.ce_ser = ce_ser   removed from constructor
        self.single_CE = single_ce
        self.rand_prob = rand_prob
        self.ce_sgn = ce_sgn

    # TODO: create seperate functions for every utility function
    # TODO 2:  put them all in a list
    # TODO 3: create a function that pulls a function out of the list and passes in params for each function

    def update_q_table(self, prev_state, action, curr_state, reward):
        old_q = self.q_table[prev_state][action]
        next_q = self.q_table[curr_state, :, :]
        new_q = np.zeros(self.num_objectives)
        for o in range(self.num_objectives):
            new_q[o] = old_q[o] + self.alpha * (reward[o] + self.gamma * max(next_q[:, o]) - old_q[o])
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

    # this is the objective function to be minimised by the nonlinear optimiser
    # (therefore it returns the negative of SER)
    @abstractmethod
    def objective(self, strategy):
        pass

    @abstractmethod
    def calc_mixed_strategy_nonlinear(self, state):
        pass

    @abstractmethod
    def calc_returns(self, agent, vector):
        pass

    # Calculates the expected payoff vector for a given strategy using the agent's own Q values
    @abstractmethod
    def calc_expected_vec(self, state, strategy):
        pass

    @staticmethod
    def select_recommended_action(state):
        return state


def softmax(q):
    soft_q = np.exp(q - np.max(q))
    return soft_q / soft_q.sum(axis=0)
