"""haakon8855"""

import random
from time import time

from critic import Critic
from actor import Actor


class ReinforcementLearning:
    """
    Reinforcement learning class for handling general practicalities and
    communication between actor-critic and the sim-world.
    """

    def __init__(self, sim_world, episodes, max_steps, table_critic, epsilon,
                 lrate, trace_decay, drate):
        self.episodes = episodes
        self.max_steps = max_steps
        self.table_critic = table_critic
        self.epsilon = epsilon
        self.epsilon_d = epsilon / 10
        self.lrate = lrate  # alpha
        self.drate = drate  # gamma
        self.trace_decay = trace_decay  # lambda
        # Initialize critic, actor and sim world
        self.critic = Critic(drate=drate)
        self.actor = Actor()
        self.sim_world = sim_world

    def train(self):
        """
        Runs through episodes in order to train the basic RL model.
        """
        start_time = time()
        # Run for self.episodes number of times, printing progress every 10%
        for j in range(10):
            self.epsilon -= self.epsilon_d
            for _ in range(self.episodes // 10):
                self.one_episode()
                self.sim_world.store_game_length()
            print(j, end="")
            # self.epsilon /= j + 1
        end_time = time()

        print(f"Time spent training: {end_time-start_time}")
        self.sim_world.plot_historic_game_length()

        # Set epsilon to 0 for actual gameplay without exploration
        self.epsilon = 0
        for _ in range(10):
            self.one_episode()
            self.sim_world.plot_history()

    def one_episode(self):
        """
        Does one episode.
        """
        history = []
        state = self.sim_world.produce_initial_state()
        action = self.get_action(state)
        # Reset eligibility
        self.actor.initiate_eligibility()
        self.critic.initiate_eligibility()
        # For each step of the episode:
        end_state = False
        while not end_state:
            # 1. Do action a from state s:
            reward = self.sim_world.update(action)
            new_state = self.sim_world.get_current_state()
            # Store state-action-pair in history
            history.append((*state, action))
            # 2.
            proposed_action = self.get_action(new_state)
            # 3.
            self.actor.set_state_action_eligibility((*state, action), 1)
            # 4.
            td_error = self.critic.get_td_error(reward, state, new_state)
            # 5.
            self.critic.set_state_eligibility(state, 1)
            # 6.
            for state_action_pair in history:
                state = state_action_pair[:-1]
                action = state_action_pair[-1]
                # a
                new_state_value = self.critic.get_state_value(
                    state
                ) + self.lrate * td_error * self.critic.get_state_eligibility(
                    state)
                self.critic.set_state_value(state, new_state_value)
                # b
                new_state_eligibility = (
                    self.drate * self.trace_decay *
                    self.critic.get_state_eligibility(state))
                self.critic.set_state_eligibility(state, new_state_eligibility)
                # c
                new_state_action_value = self.actor.get_state_action_value(
                    state_action_pair
                ) + self.lrate * td_error * self.actor.get_state_action_eligibility(
                    state_action_pair)
                self.actor.set_state_action_value(state_action_pair,
                                                  new_state_action_value)
                # d
                new_state_action_eligibility = (
                    self.drate * self.trace_decay *
                    self.actor.get_state_action_eligibility(state_action_pair))
                self.actor.set_state_action_eligibility(
                    state_action_pair, new_state_action_eligibility)
            # 7.
            state = new_state
            action = proposed_action
            # 8. Check if state is final or failed state
            if (self.sim_world.is_current_state_failed_state()
                    or self.sim_world.is_current_state_final_state()):
                end_state = True

    def get_action(self, state):
        """
        Returns an action given a state by consulting the actor
        """
        do_argmax = random.random() > self.epsilon
        possible_actions = self.sim_world.get_legal_actions(state)
        return self.actor.get_proposed_action(do_argmax, state,
                                              possible_actions)
