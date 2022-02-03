"""Haakoas"""

from random import uniform
import numpy as np


class PoleBalancing():
    """
    PoleBalancing class for holding the simulated world for balancing a pole
    on a cart.
    """
    def __init__(self):
        # Constants:
        self.length = 0.5  # m
        self.mass_p = 0.1  # kg
        self.mass_c = 1  # kg
        self.gravity = 9.8  # m/s^2
        self.force = 10  # N
        self.max_angle = 0.21  # radians
        self.max_x_pos = 2.4  # m
        self.tau = 0.02  # s, timestep length tau
        self.steps = 300  # num of timesteps in episode
        # State parameters:
        self.angle = 0
        self.angle_vel = 0
        self.x_pos = 0
        self.x_vel = 0
        self.current_step = 0
        self.balancing_failed = False
        self.cart_exited = False
        self.produce_initial_state()

    def produce_initial_state(self):
        """
        Initializes the sim world to its initial state where the cart has
        zero velocity, the pole has zero angular velocity, the cart is placed
        in the middle of the world and the angle of the pole is randomized
        between the lower and upper bound for the angle.
        """
        self.angle = uniform(-self.max_angle, self.max_angle)
        self.angle_vel = 0
        self.x_pos = 0
        self.x_vel = 0
        self.current_step = 0
        self.balancing_failed = False
        self.cart_exited = False
        return self.get_current_state()

    def update(self, action: bool):
        """
        Advances the sim world by one timestep.
        Parameter means to apply F if True and -F if False, i.e. either go right
        or go left.
        """
        self.current_step += 1
        next_state = self.get_child_state(action)
        self.x_pos = next_state[0]
        self.x_vel = next_state[1]
        self.angle = next_state[2]
        self.angle_vel = next_state[3]

        # Update state values with the newly updated ones
        if not self.balancing_failed:
            if np.abs(self.angle) > self.max_angle:
                self.balancing_failed = True
        if not self.cart_exited:
            if np.abs(self.x_pos) > self.max_x_pos:
                self.cart_exited = True
        return 1  # Reward

    def get_child_state(self, action: bool, rounded=False):
        """
        Returns the child state if given action is performed.
        """
        # Set the bangbang-force, either positive or negative F
        bb_force = [-self.force, self.force][action]
        # Calculate double derivatives
        angle_acc = self.update_angle_acc(bb_force)
        x_acc = self.update_x_acc(bb_force, angle_acc)
        # Calculate state variables
        x_pos = self.x_pos + self.tau * self.x_vel
        x_vel = self.x_vel + self.tau * x_acc
        angle = self.angle + self.tau * self.angle_vel
        angle_vel = self.angle_vel + self.tau * angle_acc
        if rounded:
            return round(x_pos, 2), round(x_vel,
                                          2), round(angle,
                                                    2), round(angle_vel, 2)
        return x_pos, x_vel, angle, angle_vel

    def update_angle_acc(self, bb_force):
        """
        Updates the acceleration of the angle.
        """
        numerator_fraction = (np.cos(self.angle) *
                              (-bb_force - self.mass_p * self.length *
                               (self.angle_vel**2) * np.sin(self.angle))) / (
                                   self.mass_p + self.mass_c)
        numerator = self.gravity * np.sin(self.angle) + numerator_fraction
        denominator = self.length * (4 / 3 - (self.mass_p *
                                              (np.cos(self.angle)**2)) /
                                     (self.mass_p + self.mass_c))
        return numerator / denominator

    def update_x_acc(self, bb_force, angle_acc):
        """
        Updates the acceleration of the cart.
        """
        numerator = bb_force + self.mass_p * self.length * (
            (self.angle_vel**2) * np.sin(self.angle) -
            angle_acc * np.cos(self.angle))
        denominator = self.mass_p + self.mass_c
        return numerator / denominator

    def get_current_state(self):
        """
        Returns the current state of the sim world.
        """
        return round(self.x_pos,
                     2), round(self.x_vel,
                               2), round(self.angle,
                                         2), round(self.angle_vel, 2)

    def is_current_state_final_state(self):
        """
        Returns whether the current state is a final state.
        Returns True if current step is more than the number of steps to success
        so long as the cart never exited the area during the entire episode and
        as long as the pole never went outside the permitted angle range during
        the entire episode.
        """
        return (self.current_step >= self.steps) and (
            not self.cart_exited) and (not self.balancing_failed)

    def get_legal_actions(self):
        """
        Returns the legal actions from the current state. The action (boolean)
        represents whether the cart will be pushed to the right or not. If True
        it will be pushed to the right, if False it will be pushed to the left.
        """
        return False, True

    def get_child_states(self, rounded=True):
        """
        Returns the child states from the current state as a dictionary where
        the key is a possible action and the corresponding value is the
        resulting state from that action.
        """
        child_states = {}
        for legal_action in self.get_legal_actions():
            child_states[legal_action] = self.get_child_state(
                legal_action, rounded)
        return child_states

    def __str__(self):
        outstring = ""
        outstring += f"\nx_pos: {self.x_pos}"
        outstring += f"\nx_vel: {self.x_vel}"
        outstring += f"\nangle: {self.angle}"
        outstring += f"\nangle_vel: {self.angle_vel}"
        return outstring


if __name__ == "__main__":
    pole = PoleBalancing()
    for _ in range(10):
        pole.update(False)
        print(str(pole))
    # pole.update(False)
    # print(str(pole))
    # print(pole.get_child_states())
