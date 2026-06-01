import casadi as ca
import numpy as np
from utils import ControllerBase


class CEC(ControllerBase):
    name = "cec"
    
    def __init__(self, time_step, *args, **kwargs) -> None:
        super().__init__(time_step)

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        # TODO: define optimization variables

        # TODO: define optimization constraints and optimization objective

        # TODO: define optimization solver
        nlp = ...
        solver = ca.nlpsol("S", "ipopt", nlp)
        sol = solver(
            x0=...,  # TODO: initial guess
            lbx=..., # TODO: lower bound on optimization variables
            ubx=..., # TODO: upper bound on optimization variables
            lbg=..., # TODO: lower bound on optimization constraints
            ubg=..., # TODO: upper bound on optimization constraints
        )
        x = sol["x"]  # get the solution

        # TODO: extract the control input from the solution
        u = ...
        return u
    
    def _car_error_kinematics(self, cur_err_state, control, cur_ref_state, next_ref_state):
        """
        Compute the error kinematics given the current error state and control input.
        Args:
            cur_err_state: current error state
            control: control input
            cur_ref_state: current reference state
            next_ref_state: next reference state
        Returns:
            next_err_state: next error state
        """
        theta = cur_err_state[2] + cur_ref_state[2]
        d_phi = control[1] * self.dt / 2  # half step rotation (w*dt/2)
        phi = theta + d_phi
        rot_3d_z_exact_integration = ca.vertcat(
            ca.horzcat(self.dt * ca.sinc(d_phi) * ca.cos(phi), 0),
            ca.horzcat(self.dt * ca.sinc(d_phi) * ca.sin(phi), 0),
            ca.horzcat(0, self.dt)
        )
        F = rot_3d_z_exact_integration @ control

        ref_err = cur_ref_state - next_ref_state

        next_err_state = cur_err_state + F + ref_err
        next_err_state[2] = ca.arctan2(ca.sin(next_err_state[2]), ca.cos(next_err_state[2]))

        return next_err_state

