"""haakon8855"""

from configuration import Config
from reinforcement_learning import ReinforcementLearning
from pole_balancing import PoleBalancing
from hanoi import Hanoi


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
        self.lrate = float(conf_globals['lrate'])
        self.trace_decay = float(conf_globals['trace_decay'])
        self.drate = float(conf_globals['drate'])

        if self.problem == 'cartpole':
            self.sim_world = PoleBalancing()
        elif self.problem == 'hanoi':
            num_pegs = int(conf_globals['num_pegs'])
            num_discs = int(conf_globals['num_discs'])
            self.sim_world = Hanoi(num_pegs=num_pegs, num_discs=num_discs)

        self.reinforcement_learner = ReinforcementLearning(
            self.sim_world, self.episodes, self.max_steps, self.table_critic,
            self.epsilon, self.lrate, self.trace_decay, self.drate)

    def run(self):
        """
        Runs the reinforcement learning system on the specified problem/simworld
        """
        self.reinforcement_learner.train()


if __name__ == "__main__":
    # gprl = GPRLSystem("config_pole.ini")
    gprl = GPRLSystem("config_hanoi.ini")
    gprl.run()
