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

    def __init__(self,
                 table_critic,
                 lrate,
                 drate,
                 trace_decay,
                 seed=None,
                 nn_dims=None):
        self.state_value = defaultdict(Critic.default_state_value)
        self.state_eligibility = defaultdict(lambda: 0)
        self.table_critic = table_critic
        self.state_value_nn = None
        self.lrate = lrate
        self.drate = drate
        self.trace_decay = trace_decay

        # Initiate the dimensions of the neural network
        self.nn_dims = nn_dims
        if self.nn_dims is None:
            self.nn_dims = [50, 1]

        # Seed the RNG if seed is specified
        if seed is not None:
            ks.utils.set_random_seed(seed)

        # Initiate the neural network if the critic is not table-based
        if not table_critic:
            self.init_neural_network()

    def init_neural_network(self):
        """
        Initializes the neural network.
        """
        opt = ks.optimizers.Adam  # NN optimizer
        model = ks.models.Sequential()  # Init base model

        # Populate layers
        for nodes in self.nn_dims[:-1]:
            model.add(ks.layers.Dense(nodes, activation='tanh'))
        model.add(ks.layers.Dense(self.nn_dims[-1]))  # Output layer
        model.compile(optimizer=opt(learning_rate=self.lrate), loss='mse')
        # Store model reference
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
        # Use table or neural net depending on config parameter
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
        Only for table based critic:
        Update the state evaluation given the current state and td_error.
        """
        # Calculates the new state-value
        new_state_value = self.get_state_value(
            state) + self.lrate * td_error * self.get_state_eligibility(state)
        # Stores it in the table
        self.set_state_value(state, new_state_value)

    def update_state_values(self, states, targets):
        """
        Only for NN based critic:
        Update the state evaluations given a list of states and td_error.
        """
        self.state_value_nn.fit(states, targets, epochs=10, verbose=0)

    def update_state_eligibility(self, state):
        """
        Only for table based critic:
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
