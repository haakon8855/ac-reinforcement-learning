"""haakon8855"""

from matplotlib import pyplot as plt
import numpy as np
# import cProfile, pstats
# import io

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
        self.seed = None
        # TODO: Define nn in config
        if 'seed' in conf_globals:
            self.seed = int(conf_globals['seed'])

        if self.problem == 'cartpole':
            self.sim_world = PoleBalancing()
        elif self.problem == 'hanoi':
            num_pegs = int(conf_globals['num_pegs'])
            num_discs = int(conf_globals['num_discs'])
            self.sim_world = Hanoi(num_pegs=num_pegs, num_discs=num_discs)
        elif self.problem == 'gambler':
            win_prob = float(conf_globals['win_prob'])
            self.sim_world = Gambler(win_prob=win_prob)

        self.reinforcement_learner = ReinforcementLearning(
            self.sim_world, self.episodes, self.max_steps, self.table_critic,
            self.epsilon, self.actor_lrate, self.critic_lrate,
            self.trace_decay, self.drate, self.verbose, self.seed)

        if self.problem == 'gambler':
            self.before = True
            self.visualize_gambler_policy()

    def run(self):
        """
        Runs the reinforcement learning system on the specified problem/simworld
        """
        self.reinforcement_learner.train()
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
        # Draw stars where peaks are expected
        # for i in range(1, 8):
        #     # plt.axvline(12.5 * i, color='tab:gray', linestyle='--')
        #     plt.scatter(12.5 * i,
        #                 50 - 12 * abs(i - 4),
        #                 s=200,
        #                 marker=(5, 1),
        #                 color='tab:gray')
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
    # profiler = cProfile.Profile()
    # profiler.enable()

    # gprl = GPRLSystem("configs/config_pole.ini")
    gprl = GPRLSystem("configs/config_hanoi.ini")
    # gprl = GPRLSystem("configs/config_gambler.ini")
    # gprl = GPRLSystem("configs/config_pole_nn.ini")
    # gprl = GPRLSystem("configs/config_hanoi_nn.ini")
    # gprl = GPRLSystem("configs/config_gambler_nn.ini")
    gprl.run()

    # profiler.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    # ps.print_stats()
    # with open('test.txt', 'w+') as f:
    #     f.write(s.getvalue())


if __name__ == "__main__":
    main()
