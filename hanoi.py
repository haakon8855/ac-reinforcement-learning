"""haakon8855"""

from matplotlib import pyplot as plt


class Hanoi:
    """
    Hanoi class for holding the simulated world of 'Towers of Hanoi'.
    """

    def __init__(self, num_pegs=3, num_discs=3):
        # Constants:
        self.num_pegs = num_pegs
        self.num_discs = num_discs
        # State parameters:
        self.state = [0 for _ in range(num_discs)]
        self.current_step = 0
        self.max_steps = 300
        self.failed = False
        self.history = []
        self.historic_game_length = []
        self.possible_actions = []
        self.calculate_possible_actions()
        self.produce_initial_state()

    def calculate_possible_actions(self):
        """
        Populates the list of all possible moves. (Disregarding state and
        illegal moves.)
        """
        for i in range(self.num_pegs):
            for j in range(self.num_pegs):
                if i != j:
                    self.possible_actions.append((i, j))

    def produce_initial_state(self):
        """
        Initializes the sim world to its initial state where all blocks are
        placed on the leftmost pole.
        """
        self.current_step = 0
        self.state = [0 for _ in range(self.num_discs)]
        self.failed = False
        self.history = []
        return self.get_current_state()

    def update(self, action: int):
        """
        Advances the sim world by one step. The action maps to the index in
        the list of all possible actions, containing tuples on the form
        (source pole index, target pole index).
        """
        self.current_step += 1

        if not self.action_is_legal(action):
            raise Exception("Illegal action")
        self.state = self.get_child_state(action)

        self.history.append(self.state)

        # Update state values with the newly updated ones
        if not self.failed:
            if self.current_step >= self.max_steps:
                self.failed = True

        # Return reward
        reward = -1
        return reward

    def get_child_state(self, action: int):
        """
        Returns the child state if given action is performed.
        Does not change the world's state.
        """
        state = self.state
        action_indexes = self.possible_actions[action]
        for i in range(len(state) - 1, -1, -1):
            if state[i] == action_indexes[0]:
                state[i] = action_indexes[1]
                break
        return state

    def get_current_state(self):
        """
        Returns the current state of the sim world.
        """
        oh_state = Hanoi.one_hot_state(self.state, self.num_pegs)
        # return (*self.state, )
        return tuple(oh_state)

    def is_current_state_final_state(self):
        """
        Returns whether the current state is a final state.
        Returns True if all blocks are on the rightmost pole.
        """
        for disc in self.state:
            if disc != self.num_pegs - 1:
                return False
        return True

    def is_current_state_failed_state(self):
        """
        Returns whether the current state is a failed state or not.
        Returns True only if maximum number of turns is reached (i.e. timeout).
        """
        return self.failed

    def get_legal_actions(self, state=None):
        """
        Returns a list of legal actions in the current state. The action (int)
        is an index mapping to the list of all possible moves.
        """
        if state is None:
            state = self.state
        legal_actions = []
        for i in range(len(self.possible_actions)):
            if self.action_is_legal(i):
                legal_actions.append(i)
        return legal_actions

    def action_is_legal(self, action):
        """
        Returns whether the given action is legal to perform or not from the
        current state.
        """
        for i in range(len(self.state) - 1, -1, -1):
            if self.state[i] == self.possible_actions[action][1]:
                return False
            if self.state[i] == self.possible_actions[action][0]:
                return True

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

    @staticmethod
    def one_hot_state(state, num_pegs):
        """
        Returns the one-hot encoding of the state representatinon given.
        """
        oh_state = []
        for pos in state:
            oh_state += Hanoi.one_hot_variable(pos, num_pegs)
        return oh_state

    @staticmethod
    def one_hot_variable(disc_pos: int, num_pegs: int):
        """
        Returns the one-hot encoding of one rounded state variable
        """
        vector = [0] * (num_pegs)
        vector[disc_pos] = 1
        return vector


if __name__ == "__main__":
    hanoi = Hanoi()
    print(str(hanoi))
    print(hanoi.get_legal_actions())
    print(hanoi.possible_actions)
    hanoi.update(1)
    print(str(hanoi))
    print(hanoi.get_legal_actions())
    # pole.update(False)
    # print(str(pole))
    # print(pole.get_child_states())
