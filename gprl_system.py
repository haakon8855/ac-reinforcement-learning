"""haakon8855"""

import json
from matplotlib import pyplot as plt
import numpy as np

from configuration import Config
from reinforcement_learning import ReinforcementLearning
from pole_balancing import PoleBalancing
from hanoi import Hanoi
from gambler import Gambler


class GPRLSystem:
    """
    General purpose reinforcement learning system
    """

    def __init__(self, config_file: str):
        # Fetching configuration parameters from given config file
        self.config = Config.get_config(config_file)
        conf_globals = self.config['GLOBALS']
        self.problem = conf_globals['problem']
        self.episodes = int(conf_globals['episodes'])
        self.max_steps = int(conf_globals['max_steps'])
        self.table_critic = conf_globals['table_critic'] == 'true'
        self.epsilon = float(conf_globals['epsilon'])
        self.actor_lrate = float(conf_globals['actor_lrate'])
        self.critic_lrate = float(conf_globals['critic_lrate'])
        self.trace_decay = float(conf_globals['trace_decay'])
        self.drate = float(conf_globals['drate'])
        self.verbose = conf_globals['verbose'] == 'true'

        # If a seed is specified in the config, we will set the random seed
        self.seed = None
        if 'seed' in conf_globals:
            self.seed = int(conf_globals['seed'])

        # If the critic is neural-network-based, we fetch the network's shape
        self.network_dimensions = None
        if not self.table_critic:
            self.network_dimensions = json.loads(conf_globals['nn_dims'])

        # Fetch parameters specific to the cartpole problem and create
        # an instance of the simworld.
        if self.problem == 'cartpole':
            self.length = float(conf_globals['length'])
            self.mass_p = float(conf_globals['pole_mass'])
            self.gravity = float(conf_globals['gravity'])
            self.tau = float(conf_globals['timestep'])
            self.sim_world = PoleBalancing(self.length,
                                           self.mass_p,
                                           self.gravity,
                                           self.tau,
                                           max_steps=self.max_steps)
        # Fetch parameters specific to the ToH problem and create
        # an instance of the simworld.
        elif self.problem == 'hanoi':
            num_pegs = int(conf_globals['num_pegs'])
            num_discs = int(conf_globals['num_discs'])
            anim_delay = 0.5
            if 'anim_delay' in conf_globals:
                anim_delay = float(conf_globals['anim_delay'])
            self.sim_world = Hanoi(num_pegs=num_pegs,
                                   num_discs=num_discs,
                                   animation_delay=anim_delay,
                                   max_steps=self.max_steps)
        # Fetch parameters specific to the gambler problem and create
        # an instance of the simworld.
        elif self.problem == 'gambler':
            win_prob = float(conf_globals['win_prob'])
            self.sim_world = Gambler(win_prob=win_prob)

        # Create the reinforcement learner instance, passing necessary params
        self.reinforcement_learner = ReinforcementLearning(
            self.sim_world, self.episodes, self.max_steps, self.table_critic,
            self.epsilon, self.actor_lrate, self.critic_lrate,
            self.trace_decay, self.drate, self.verbose, self.seed,
            self.network_dimensions)

        # Run visualization of the gambler policy before training if current
        # run solves the gambler problem.
        if self.problem == 'gambler':
            self.before = True
            self.visualize_gambler_policy()

    def run(self):
        """
        Runs the reinforcement learning system on the specified problem/simworld
        """
        self.reinforcement_learner.train()
        # Run visualization of the gambler policy after training if current
        # run solves the gambler problem.
        if self.problem == 'gambler':
            self.visualize_gambler_policy()

    def visualize_gambler_policy(self):
        """
        Visualizes the policy for the gambler simworld.
        """
        min_state = 1
        max_state = self.sim_world.max_coins
        states_xaxis = list(range(min_state, max_state))
        states = np.eye(max_state + 1, dtype=int)[1:-1]
        wagers = []
        for state in states:
            wagers.append(self.reinforcement_learner.get_action(tuple(state)))
        plt.plot(states_xaxis, wagers)

        if self.before:
            plt.savefig('plots/before.png')
            self.before = False
        else:
            plt.savefig('plots/after.png')
        plt.clf()


def main():
    """
    Main function for running this python script.
    """

    # gprl = GPRLSystem("configs/config_pole.ini")
    gprl = GPRLSystem("configs/config_hanoi.ini")
    # gprl = GPRLSystem("configs/config_gambler.ini")
    # gprl = GPRLSystem("configs/config_pole_nn.ini")
    # gprl = GPRLSystem("configs/config_hanoi_nn.ini")
    # gprl = GPRLSystem("configs/config_gambler_nn.ini")
    gprl.run()


if __name__ == "__main__":
    main()
