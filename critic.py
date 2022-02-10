"""haakon8855"""

from collections import defaultdict
from random import random


class Critic:
    """
    Critic class for housing the critic which will give feedback to the actor
    on its actions.
    """

    def __init__(self, lrate, drate, trace_decay):
        self.state_value = defaultdict(Critic.default_state_value)
        self.state_eligibility = defaultdict(lambda: 0)
        self.lrate = lrate
        self.drate = drate
        self.trace_decay = trace_decay

    def get_td_error(self, reward, state, new_state):
        """
        Returns the td_error given a reward, a state and the next state.
        """
        return reward + self.drate * self.get_state_value(
            new_state) - self.get_state_value(state)

    def get_state_value(self, state):
        """
        Returns the value of a given state.
        """
        return self.state_value[state]

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

    def update_state_eligibility(self, state):
        """
        Update the state eligibility given the current state.
        """
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
