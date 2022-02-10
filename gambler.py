"""haakon8855"""

from random import randint, random
from matplotlib import pyplot as plt


class Gambler:
    """
    Gambler class for holding the simulated world of the gambler.
    """

    def __init__(self, win_prob=0.4):
        # Constants:
        self.max_coins = 100
        self.min_bet = 1

        # State parameters:
        self.state = 0
        self.win_prob = win_prob
        self.current_step = 0
        self.max_steps = 300
        self.failed = False
        self.history = []
        self.historic_game_length = []
        self.possible_actions = []

        # Initialization
        self.produce_initial_state()

    def produce_initial_state(self):
        """
        Initializes the sim world to its initial state with a random
        amount of coins.
        """
        self.current_step = 0
        self.state = randint(1, 99)
        self.failed = False
        self.history = [self.state]
        return self.get_current_state()

    def update(self, action: int):
        """
        Advances the sim world by one step. The action (int) represents the
        amount wagered.
        """
        if self.failed:
            return -1000

        self.current_step += 1

        if not self.action_is_legal(action):
            raise Exception("Illegal action")

        old_state = self.state
        self.state = self.get_child_state(action)

        self.history.append(self.state)

        # Check if state is failed state
        if not self.failed:
            if self.current_step >= self.max_steps or self.state == 0:
                self.failed = True

        # Return reward
        reward = self.state - old_state
        return reward

    def get_child_state(self, action: int):
        """
        Returns a child state if given action is performed.
        Does not change the world's state.
        """
        state = self.state
        if random() < self.win_prob:
            state += action  # Win money
        else:
            state -= action  # Lose money
        return state

    def get_current_state(self):
        """
        Returns the current state of the sim world.
        """
        oh_state = [0] * (self.max_coins + 1)
        oh_state[self.state] = 1
        return tuple(oh_state)
        return (self.state, )

    def is_current_state_final_state(self):
        """
        Returns whether the current state is a final state.
        Returns True if number of coins is self.max_coins.
        """
        return self.state == self.max_coins

    def is_current_state_failed_state(self):
        """
        Returns whether the current state is a failed state or not.
        Returns True if maximum number of turns is reached (i.e. timeout)
        or number of coins is 0.
        """
        return self.failed

    def get_legal_actions(self, state=None):
        """
        Returns a list of legal actions in the current state. The action (int)
        represents the number of coins to wager.
        """
        if state is None:
            state = self.state
        else:
            state = state.index(1)
        dist_to_win = self.max_coins - state
        dist_to_lose = state
        max_bet = min(dist_to_lose, dist_to_win)
        return [self.min_bet] + list(range(self.min_bet + 1, max_bet + 1))

    def action_is_legal(self, action):
        """
        Returns whether the given action is legal to perform or not from the
        current state.
        """
        dist_to_win = self.max_coins - self.state
        dist_to_lose = self.state
        max_bet = min(dist_to_lose, dist_to_win)
        return action >= self.min_bet and action <= max_bet

    def plot_history(self):
        """
        Plots the course of the current game up until current state.
        """
        # TODO: Do

    def plot_historic_game_length(self):
        """
        Plots the number of steps used in each historic game.
        """
        plt.plot(self.historic_game_length)
        plt.show()

    def store_game_length(self):
        """
        Stores the game length in a list to plot later.
        """
        self.historic_game_length.append(self.current_step)

    def __str__(self):
        outstring = f"state: {self.state}"
        return outstring


if __name__ == "__main__":
    hanoi = Gambler()
    print(str(hanoi))
    print(hanoi.get_legal_actions())
    print(hanoi.possible_actions)
    hanoi.update(1)
    print(str(hanoi))
    print(hanoi.get_legal_actions())
    # pole.update(False)
    # print(str(pole))
