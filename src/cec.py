from dataclasses import dataclass
import casadi as ca
import numpy as np
from utils import ControllerBase, ConfigBase
from utils import *


def casadi_sinc(x):
    return ca.if_else(ca.fabs(x) < 1e-9, 1.0, ca.sin(x) / x)

@dataclass(kw_only=True)
class CECConfig(ConfigBase):
    T: int
    Q: np.ndarray
    q: float = 1.0
    R: np.ndarray
    terminal_Q: np.ndarray
    terminal_q: float = 1.0
    gamma: float

class CEC(ControllerBase):
    name = "cec"
    
    def __init__(self, config: CECConfig, *args, **kwargs) -> None:
        super().__init__(config)

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
        # define optimization variables
        e = ca.SX.sym("e", 3, self.config.T + 1)
        u = ca.SX.sym("u", 2, self.config.T)

        lb = []
        ub = []
        # error state bounds
        for i in range(self.config.T + 1):
            lb.extend([-np.inf, -np.inf, -np.pi])
            ub.extend([np.inf, np.inf, np.pi])
        # control bounds
        for i in range(self.config.T):
            lb.extend([V_MIN, W_MIN])
            ub.extend([V_MAX, W_MAX])

        # define optimization constraints
        g = []
        lbg = []
        ubg = []

        # compute reference trajectory
        ref_traj = self._compute_reference_trajectory(t)
        cur_err_state = cur_state - cur_ref_state
        cur_err_state[2] = CEC._angle_wrap(cur_err_state[2])

        # initial error state constraint
        g.append(e[:, 0] - cur_err_state)
        lbg.extend([0.0, 0.0, 0.0])
        ubg.extend([0.0, 0.0, 0.0])
        
        # error kinematics constraints
        for i in range(self.config.T):
            next_err_state = self._car_error_kinematics(e[:, i], u[:, i], ref_traj[i], ref_traj[i + 1])
            g.append(e[:, i + 1] - next_err_state)
            lbg.extend([0.0, 0.0, 0.0])
            ubg.extend([0.0, 0.0, 0.0])

        # add collision avoidance constraints
        for i in range(self.config.T + 1):
            ref_state = ref_traj[i]
            state = e[:, i] + ref_state
            g.append(state[0])
            lbg.append(WORLD_BOUNDS[0])
            ubg.append(WORLD_BOUNDS[1])
            g.append(state[1])
            lbg.append(WORLD_BOUNDS[2])
            ubg.append(WORLD_BOUNDS[3])
            g.append(ca.cos(state[2]))
            lbg.append(-1.0)
            ubg.append(1.0)
            g.append(ca.sin(state[2]))
            lbg.append(-1.0)
            ubg.append(1.0)

            if len(self.cspace_obstacles) > 0:
                for obs in self.cspace_obstacles:
                    obs_x, obs_y, obs_r = obs
                    dist_sq = (state[0] - obs_x) ** 2 + (state[1] - obs_y) ** 2
                    g.append(dist_sq)
                    lbg.append(obs_r ** 2)
                    ubg.append(np.inf)

        # define the optimization objective
        obj = 0
        for i in range(self.config.T):
            state_cost = ca.mtimes([e[:2, i].T, self.config.Q, e[:2, i]]) + self.config.q * (1 - ca.cos(e[2, i])) ** 2  # angle error cost
            control_cost = ca.mtimes([u[:, i].T, self.config.R, u[:, i]])
            obj += self.config.gamma ** i * (state_cost + control_cost)                                                 # discount factor
        obj += ca.mtimes([e[:2, self.config.T].T, self.config.terminal_Q, e[:2, self.config.T]]) \
            + self.config.terminal_q * (1 - ca.cos(e[2, self.config.T])) ** 2                                           # terminal cost

        # define optimization solver
        nlp = {
            'x': ca.vertcat(ca.reshape(e, -1, 1), ca.reshape(u, -1, 1)),
            'f': obj,
            'g': ca.vertcat(*g)
        }
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
        }
        solver = ca.nlpsol("S", "ipopt", nlp, opts)
        sol = solver(
            x0=ca.vertcat(ca.reshape(cur_err_state.tolist() * (self.config.T + 1), -1, 1), ca.reshape([0.0] * 2 * self.config.T, -1, 1)),
            lbx=ca.DM(lb),
            ubx=ca.DM(ub),
            lbg=ca.DM(lbg),
            ubg=ca.DM(ubg),
        )
        x_opt = sol["x"].full().flatten()  # get the solution

        # extract the control input from the solution
        u = x_opt[-2 * self.config.T:].reshape(-1, 2).T
        return u[:, 0]
    
    def _compute_reference_trajectory(self, t):
        """
        Compute the reference trajectory over the horizon given the current reference state.
        Args:
            t: current time step
        Returns:
            ref_traj: reference trajectory over the horizon
        """
        ref_traj = []
        for i in range(self.config.T + 1):
            ref_traj.append(self.config.traj_func(t + i))
        return np.array(ref_traj)
    
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
        d_phi = control[1] * self.config.dt / 2  # half step rotation (w*dt/2)
        phi = theta + d_phi
        rot_3d_z_exact_integration = ca.vertcat(
            ca.horzcat(self.config.dt * casadi_sinc(d_phi) * ca.cos(phi), 0),
            ca.horzcat(self.config.dt * casadi_sinc(d_phi) * ca.sin(phi), 0),
            ca.horzcat(0, self.config.dt)
        )
        F = rot_3d_z_exact_integration @ control

        ref_err = cur_ref_state - next_ref_state

        next_err_state = cur_err_state + F + ref_err
        next_err_state[2] = CEC._angle_wrap(next_err_state[2])

        return next_err_state
    
    @staticmethod
    def _angle_wrap(angle):
        """
        Wrap the angle to [-pi, pi].
        Args:
            angle: input angle
        Returns:
            wrapped_angle: wrapped angle
        """
        return ca.arctan2(ca.sin(angle), ca.cos(angle))

