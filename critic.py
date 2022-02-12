"""haakon8855"""

from collections import defaultdict
from random import random
from tensorflow import keras as ks
import numpy as np


class Critic:
    """
    Critic class for housing the critic which will give feedback to the actor
    on its actions.
    """

    def __init__(self, table_critic, state_length, lrate, drate, trace_decay):
        self.state_value = defaultdict(Critic.default_state_value)
        self.state_eligibility = defaultdict(lambda: 0)
        self.table_critic = table_critic
        self.state_value_nn = None
        self.lrate = lrate
        self.drate = drate
        self.trace_decay = trace_decay
        if not table_critic:
            self.init_neural_network(state_length)

    def init_neural_network(self, state_length):
        """
        Initializes the neural network.
        """
        opt = ks.optimizers.Adam
        loss = ks.losses.mse
        model = ks.models.Sequential()
        model.add(ks.layers.Dense(50, activation='tanh'))
        model.add(ks.layers.Dense(50, activation='tanh'))
        model.add(ks.layers.Dense(1))
        model.compile(optimizer=opt(learning_rate=self.lrate), loss='mse')
        self.state_value_nn = model

    def get_td_error(self, reward, state, new_state):
        """
        Returns the td_error given a reward, a state and the next state.
        """
        target_td = reward + self.drate * self.get_state_value(new_state)
        return target_td - self.get_state_value(state), target_td

    def get_state_value(self, state):
        """
        Returns the value of a given state.
        """
        if self.table_critic:
            return self.state_value[state]
        return self.state_value_nn(np.array(state).reshape((1, -1)))[0, 0]

    def set_state_value(self, state, value):
        """
        Returns the value of a given state.
        """
        self.state_value[state] = value

    def get_state_eligibility(self, state):
        """
        Set the eligibility for the given state.
        """
        return self.state_eligibility[state]

    def set_state_eligibility(self, state, value):
        """
        Set the eligibility for the given state.
        """
        self.state_eligibility[state] = value

    def initiate_eligibility(self):
        """
        Initiates the state eligibility.
        """
        self.state_eligibility = defaultdict(lambda: 0)

    def update_state_value(self, state, td_error):
        """
        Update the state evaluation given the current state and td_error.
        """
        new_state_value = self.get_state_value(
            state) + self.lrate * td_error * self.get_state_eligibility(state)
        self.set_state_value(state, new_state_value)

    def update_state_values(self, states, targets):
        """
        Update the state evaluations given a list of states and td_error.
        """
        self.state_value_nn.fit(states, targets, epochs=10, verbose=0)

    def update_state_eligibility(self, state):
        """
        Update the state eligibility given the current state.
        """
        if self.table_critic:
            new_state_eligibility = (self.drate * self.trace_decay *
                                     self.get_state_eligibility(state))
            self.set_state_eligibility(state, new_state_eligibility)

    @staticmethod
    def default_state_value():
        """
        Returns the default value of a dictionary object that has not been
        accessed yet.
        """
        return random() * 0.5
