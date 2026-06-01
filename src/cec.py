import casadi
import numpy as np
from utils import ControllerBase


class CEC(ControllerBase):
    name = "cec"
    
    def __init__(self) -> None:
        raise NotImplementedError

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
        solver = casadi.nlpsol("S", "ipopt", nlp)
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

