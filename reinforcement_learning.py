"""haakon8855"""

import random
from time import time
import numpy as np

from critic import Critic
from actor import Actor


class ReinforcementLearning:
    """
    Reinforcement learning class for handling general practicalities and
    communication between actor-critic and the sim-world.
    """

    def __init__(self,
                 sim_world,
                 episodes,
                 max_steps,
                 table_critic,
                 epsilon,
                 actor_lrate,
                 critic_lrate,
                 trace_decay,
                 drate,
                 verbose=False,
                 seed=None):
        self.episodes = episodes
        self.max_steps = max_steps
        self.table_critic = table_critic
        self.epsilon = epsilon
        self.epsilon_d = epsilon / 10
        self.actor_lrate = actor_lrate  # alpha
        self.critic_lrate = critic_lrate  # alpha
        self.drate = drate  # gamma
        self.trace_decay = trace_decay  # lambda
        self.verbose = verbose
        # Initialize critic, actor and sim world
        self.sim_world = sim_world
        self.critic = Critic(table_critic, critic_lrate, drate, trace_decay,
                             seed)
        self.actor = Actor(actor_lrate, drate, trace_decay)

    def train(self):
        """
        Runs through episodes in order to train the basic RL model.
        """
        start_time = time()
        train_episode = self.one_episode
        if not self.table_critic:
            train_episode = self.one_episode_nn
        # Run for self.episodes number of times, printing progress every 10%
        for j in range(10):
            for _ in range(self.episodes // 10):
                thyme = time()
                train_episode()
                if self.verbose:
                    print(round(time() - thyme, 2), end="")
                    print(f", Steps: {self.sim_world.current_step}")
                self.sim_world.store_game_length()
            print(j, end="")
            self.decrease_epsilon()
        end_time = time()

        print(f"Time spent training: {end_time-start_time}")
        self.sim_world.plot_historic_game_length()

        # Set epsilon to 0 for actual gameplay without exploration
        self.epsilon = 0
        for _ in range(10):
            self.one_episode()
            self.sim_world.plot_history()

    def decrease_epsilon(self):
        """
        Decreases epsilon.
        """
        if self.table_critic:
            self.epsilon -= self.epsilon_d

    def one_episode_nn(self):
        """
        Does one episode.
        """
        history = []
        target_history = []
        state = self.sim_world.produce_initial_state()
        action = self.get_action(state)
        # Reset eligibility
        self.actor.initiate_eligibility()
        # For each step of the episode:
        while True:
            # 1. Do action a from state s:
            reward = self.sim_world.update(action)
            new_state = self.sim_world.get_current_state()
            if self.sim_world.is_current_state_final_state():
                history.append((*new_state, 0))
                states = np.array(history)[:, :-1]
                target_history.append(0)
                self.critic.update_state_values(
                    states,
                    np.array(target_history).reshape(-1, 1))
                break
            # Store state-action-pair in history
            history.append((*state, action))
            # 2.
            proposed_action = self.get_action(new_state)
            # 3.
            self.actor.set_state_action_eligibility((*state, action), 1)
            # 4.
            td_error, target_td = self.critic.get_td_error(
                reward, state, new_state)
            target_history.append(target_td)
            # 5.
            self.critic.set_state_eligibility(state, 1)
            # 6.
            for state_action_pair in history:
                state = state_action_pair[:-1]
                action = state_action_pair[-1]
                self.critic.update_state_eligibility(state)
                self.actor.update_state_action_value(state_action_pair,
                                                     td_error)
                self.actor.update_state_action_eligibility(state_action_pair)
            # 7.
            state = new_state
            action = proposed_action
            # 8. Check if state is final or failed state
            if (self.sim_world.is_current_state_failed_state()
                    or self.sim_world.is_current_state_final_state()):
                states = np.array(history)[:, :-1]
                targets = np.array(target_history).reshape(-1, 1)
                self.critic.update_state_values(states, targets)
                break

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
            td_error, _ = self.critic.get_td_error(reward, state, new_state)
            # 5.
            self.critic.set_state_eligibility(state, 1)
            # 6.
            for state_action_pair in history:
                state = state_action_pair[:-1]
                action = state_action_pair[-1]
                self.critic.update_state_value(state, td_error)
                self.critic.update_state_eligibility(state)
                self.actor.update_state_action_value(state_action_pair,
                                                     td_error)
                self.actor.update_state_action_eligibility(state_action_pair)
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
