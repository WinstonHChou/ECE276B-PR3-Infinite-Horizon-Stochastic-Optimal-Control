from dataclasses import dataclass
import numpy as np
from value_function import GridValueFunction, FeatureValueFunction
from utils import WORLD_BOUNDS, ControllerBase
from cec import CECConfig
import utils
from numba import njit, prange


@dataclass(kw_only=True)
class GPIConfig(CECConfig):
    epos_band: list[tuple[float, float, int]]  # list of (inner, outer, n_points) triples for multiband grid
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray
    num_evals: int  # number of policy evaluations in each iteration
    collision_margin: float
    collision_cost_weight: float
    V_cls: classmethod = GridValueFunction
    output_dir: str

@dataclass(kw_only=True)
class FeatureGPIConfig(GPIConfig):
    V_cls: classmethod = FeatureValueFunction
    # used by feature-based value function
    v_ex_space: np.ndarray
    v_ey_space: np.ndarray
    v_etheta_space: np.ndarray
    v_alpha: float
    v_beta_t: float
    v_beta_e: float
    v_lr: float
    v_batch_size: int  # batch size if GPU memory is not enough


class GPI(ControllerBase):
    name = "gpi"
    
    def __init__(self, config: GPIConfig, *args, **kwargs) -> None:
        super().__init__(config)
        # Path definition and output dir setup
        self.mask_dir = utils.Path(self.config.output_dir) / "valid_err_states_mask.npy"
        self.transition_dir = utils.Path(self.config.output_dir) / "transition_matrix.npy"
        self.stage_cost_dir = utils.Path(self.config.output_dir) / "stage_costs.npy"
        self.policy_dir = utils.Path(self.config.output_dir) / "policy.npy"
        self.value_dir = utils.Path(self.config.output_dir) / "value.npy"
        self.info_dir = utils.Path(self.config.output_dir) / "info.txt"
        utils.Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize the state and control spaces from config for later use
        self.ex_space = GPI._create_multiband_grid(config.epos_band)
        self.ey_space = GPI._create_multiband_grid(config.epos_band)
        self.eth_space = config.eth_space
        self.v_space = config.v_space
        self.w_space = config.w_space

        # Initialize the reference trajectory
        self.ref_traj = self._compute_reference_trajectory()
        
        # Initialize the value function
        self.V = self.config.V_cls(self.config.T, self.ex_space, self.ey_space, self.eth_space)
        
        # Initialize dimensions of state and control spaces for later use
        self.nx, self.ny, self.nth = len(self.ex_space), len(self.ey_space), len(self.eth_space)
        self.nv, self.nw = len(self.v_space), len(self.w_space)

        # Check if dimensions are consistent with the existing caches to decide whether to reuse them or recompute
        if self.info_dir.exists():
            with open(self.info_dir, "r") as f:
                info = f.read().strip().split(",")
                if len(info) == 6:
                    prev_nx, prev_ny, prev_nth, prev_nv, prev_nw, prev_T = map(int, info)
                    if (prev_nx, prev_ny, prev_nth, prev_nv, prev_nw, prev_T) == (self.nx, self.ny, self.nth, self.nv, self.nw, self.config.T):
                        print("Existing caches are compatible with current config. Will reuse them.")
                        self.err_states_mask = self.load_mask()
                        self.transition_matrix = self.load_transition_matrix()
                        self.stage_costs = self.load_stage_costs()
                    else:
                        print("Existing caches are NOT compatible with current config. Will recompute them.")
                        self.err_states_mask = None
                        self.transition_matrix = None
                        self.stage_costs = None
                else:
                    print("Info file is corrupted. Will recompute caches.")
                    self.err_states_mask = None
                    self.transition_matrix = None
                    self.stage_costs = None
        else:
            print("No existing info file. Will compute caches.")
            self.err_states_mask = None
            self.transition_matrix = None
            self.stage_costs = None

        # Write current config info to file for future checks
        with open(self.info_dir, "w") as f:
            f.write(f"{self.nx},{self.ny},{self.nth},{self.nv},{self.nw},{self.config.T}")

        # Pre-compute the reference trajectory for the entire horizon to speed up
        if self.err_states_mask is None:
            self.err_states_mask = self._compute_valid_err_states_mask()
        if self.transition_matrix is None:
            self.transition_matrix = self._compute_transition_matrix()
        if self.stage_costs is None:
            self.stage_costs = self._compute_stage_costs()
        self.policy, _ = self.compute_policy(num_iters=100)

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
        cur_err_state = cur_state - cur_ref_state
        cur_err_state[2] = GPI._angle_wrap(cur_err_state[2])
        indices = self.state_metric_to_index(np.array([cur_err_state]))
        ex_id, ey_id, eth_id = indices[0]
        t_phase = t % self.config.T
        v = self.policy[t_phase, ex_id, ey_id, eth_id, 0]
        w = self.policy[t_phase, ex_id, ey_id, eth_id, 1]
        return np.array([v, w])
    
    def _compute_reference_trajectory(self):
        """
        Compute the reference trajectory.
        Returns:
            ref_traj: reference trajectory over the horizon
        """
        ref_traj = []
        for i in range(self.config.T):
            ref_traj.append(self.config.traj_func((i) * self.config.dt))
        return np.array(ref_traj)
    
    @utils.timer
    def _compute_valid_err_states_mask(self):
        mask = np.zeros((self.config.T, len(self.ex_space), len(self.ey_space), len(self.eth_space)), dtype=bool)
        for t in utils.tqdm(range(self.config.T)):
            ref_state = self.ref_traj[t]
            for ex_id, ex in enumerate(self.ex_space):
                for ey_id, ey in enumerate(self.ey_space):
                    for eth_id, eth in enumerate(self.eth_space):
                        err = np.array([ex, ey, eth])
                        mask[t, ex_id, ey_id, eth_id] = self._is_valid_err_state(err, ref_state)

        # save the mask to output dir
        np.save(self.mask_dir.as_posix(), mask)
        return mask

    def _is_valid_err_state(self, err_state: np.ndarray, ref_state: np.ndarray) -> bool:
        """
        Check if the error state is valid (e.g., not in collision) given the reference state.
        Args:
            err_state: error state to check
            ref_state: reference state corresponding to the error state
        Returns:
            bool: whether the error state is valid
        """
        state = err_state + ref_state

        # Check collision with world bounds
        if (state[0] < WORLD_BOUNDS[0] or 
            state[0] > WORLD_BOUNDS[1] or
            state[1] < WORLD_BOUNDS[2] or 
            state[1] > WORLD_BOUNDS[3]):
            return False

        return True

    def state_metric_to_index(self, metric_states: np.ndarray) -> np.ndarray:
        """
        Convert the metric states to grid indices according to your discretization design.
        Args:
            metric_states (np.ndarray): [N, 3] array of metric states
        Returns:
            np.ndarray: [N, 3] array of grid indices
        """
        ex_ids = GPI._find_closest_index(self.ex_space, metric_states[:, 0])
        ey_ids = GPI._find_closest_index(self.ey_space, metric_states[:, 1])
        eth_ids = GPI._find_closest_index_angle(self.eth_space, metric_states[:, 2])
        return np.stack([ex_ids, ey_ids, eth_ids], axis=1)   # (N, 3)

    def state_index_to_metric(self, state_indices: np.ndarray) -> np.ndarray:
        """
        Convert the grid indices to metric states according to your discretization design.
        Args:
            state_indices (np.ndarray): [N, 3] array of grid indices
        Returns:
            np.ndarray: [N, 3] array of metric states
        """
        ex_ids, ey_ids, eth_ids = state_indices[:, 0], state_indices[:, 1], state_indices[:, 2]
        return np.stack([self.ex_space[ex_ids], self.ey_space[ey_ids], self.eth_space[eth_ids]], axis=1)   # (N, 3)

    def control_metric_to_index(self, control_metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            control_metric: [2, N] array of controls in metric space
        Returns:
            [N, ] array of indices in the control space
        """
        v: np.ndarray = np.digitize(control_metric[0], self.v_space, right=True)
        w: np.ndarray = np.digitize(control_metric[1], self.w_space, right=True)
        return v, w

    def control_index_to_metric(self, v_ids: np.ndarray, w_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            v_ids: [N, ] array of indices in the v space
            w_ids: [N, ] array of indices in the w space
        Returns:
            [2, N] array of controls in metric space
        """
        return np.array([self.v_space[v_ids], self.w_space[w_ids]])

    @utils.timer
    def _compute_transition_matrix(self):
        """
        Compute the transition matrix in advance to speed up the GPI algorithm.
        """
        transition_matrix = np.zeros((self.config.T, self.nx, self.ny, self.nth, self.nv, self.nw, 3), dtype=int)
        for t in utils.tqdm(range(self.config.T)):
            # compute the reference state at time t and t+1 for computing the transition matrix
            cur_ref_state = self.ref_traj[t]
            next_ref_state = self.ref_traj[(t + 1) % self.config.T]  # link to the next phase for periodic infinite horizon

            # construct all combinations of current error states and control inputs for computing the transition matrix
            ex_ids, ey_ids, eth_ids, v_ids, w_ids = np.meshgrid(
                np.arange(self.nx), np.arange(self.ny), np.arange(self.nth), np.arange(self.nv), np.arange(self.nw), indexing='ij')
            cur_err_states = self.state_index_to_metric(np.stack([ex_ids.flatten(), ey_ids.flatten(), eth_ids.flatten()], axis=1))  # (N, 3)
            controls = self.control_index_to_metric(v_ids.flatten(), w_ids.flatten())  # (2, N)
            next_err_states = GPI._car_error_kinematics(cur_err_states, controls.T, cur_ref_state, next_ref_state, self.config.dt)  # (N, 3)
            transition_matrix[t] = self.state_metric_to_index(next_err_states).reshape((self.nx, self.ny, self.nth, self.nv, self.nw, 3))  # (nx, ny, nth, nv, nw, 3)
        
        # save the transition matrix to output dir
        np.save(self.transition_dir.as_posix(), transition_matrix)
        return transition_matrix

    @utils.timer
    def _compute_stage_costs(self):
        """
        Compute the stage costs in advance to speed up the GPI algorithm.
        """
        stage_costs = np.zeros((self.config.T, self.nx, self.ny, self.nth, self.nv, self.nw), dtype=float)
        for t in utils.tqdm(range(self.config.T)):
            # construct all combinations of current error states and control inputs for computing the stage costs
            ex_ids, ey_ids, eth_ids, v_ids, w_ids = np.meshgrid(
                np.arange(self.nx), np.arange(self.ny), np.arange(self.nth), np.arange(self.nv), np.arange(self.nw), indexing='ij')
            cur_err_states = self.state_index_to_metric(np.stack([ex_ids.flatten(), ey_ids.flatten(), eth_ids.flatten()], axis=1))  # (N, 3)
            controls = self.control_index_to_metric(v_ids.flatten(), w_ids.flatten())  # (2, N)
            cost = self._compute_stage_cost(cur_err_states, controls.T, self.ref_traj[t])

            # filter out invalid error states by setting their stage costs to a large number
            flattened_mask = self.err_states_mask[t].flatten()
            valid_indices = np.repeat(flattened_mask, self.nv * self.nw)
            cost[~valid_indices] = np.inf

            # Assign the computed costs to the stage_costs array
            stage_costs[t] = cost.reshape((self.nx, self.ny, self.nth, self.nv, self.nw))  # (nx, ny, nth, nv, nw)

        # save the stage costs to output dir
        np.save(self.stage_cost_dir.as_posix(), stage_costs)
        return stage_costs

    def _compute_stage_cost(self, cur_err_states: np.ndarray, controls: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
        """
        Compute the stage cost for given error states and controls.
        Args:
            cur_err_states: [N, 3] array of error states
            controls: [N, 2] array of controls
            ref_state: [3] current reference state to compute world positions

        Returns:
            cost: [N] array of stage costs
        """
        N = cur_err_states.shape[0]
        cost = np.zeros(N, dtype=float)

        Q_aug = np.broadcast_to(self.config.Q, (N, 2, 2))
        R_aug = np.broadcast_to(self.config.R, (N, 2, 2))
        
        cost += np.einsum('ni,nij,nj->n', cur_err_states[:, :2], Q_aug, cur_err_states[:, :2])  # position error cost
        cost += self.config.q * (1 - np.cos(cur_err_states[:, 2])) ** 2  # angle error cost
        cost += np.einsum('ni,nij,nj->n', controls, R_aug, controls)  # control cost

        # Compute world positions to check collisions
        cur_states = cur_err_states[:, :2] + ref_state[:2]

        # Check collision with obstacles (if any)
        collision_mask = np.zeros(N, dtype=bool)
        for obs in self.cspace_obstacles:
            obs_x, obs_y, obs_r = obs
            dist_to_center = np.linalg.norm(cur_states - np.array([obs_x, obs_y]), axis=1)
            dist = dist_to_center - obs_r
            collision_mask |= dist < self.config.collision_margin
            safe_mask = dist >= self.config.collision_margin
            # add a cost that decays exponentially with distance to obstacle (safety margin)
            cost[safe_mask] += self.config.collision_cost_weight * np.exp(-dist[safe_mask] / self.config.collision_margin)

        cost[collision_mask] = np.inf

        return cost

    def init_value_function(self):
        """
        Initialize the value function.
        """
        if self.value_dir.exists():
            self.V.value = self.load_value_function()
        else:
            self.V.value = np.zeros((self.config.T, self.nx, self.ny, self.nth))

    # def evaluate_value_function(self):
    #     """
    #     Evaluate the value function. Implement this function if you are using a feature-based value function.
    #     """
    #     # TODO: your implementation
    #     raise NotImplementedError
    
    def init_policy(self):
        """
        Initialize the policy.
        """
        if self.policy_dir.exists():
            self.policy = self.load_policy()
        else:
            self.policy = np.zeros((self.config.T, self.nx, self.ny, self.nth, 2), dtype=float)

    @utils.timer
    def policy_improvement(self):
        """
        Policy improvement step of the GPI algorithm.
        """
        policy_changed = False
        for t in utils.tqdm(range(self.config.T - 1, -1, -1)):
            next_t = (t + 1) % self.config.T  # link to the next phase for periodic infinite horizon
            new_policy, changed = self._jit_policy_improvement(
                self.stage_costs[t], 
                self.transition_matrix[t], 
                self.V.value[next_t],
                self.policy[t],
                self.config.gamma,
                self.nx, self.ny, self.nth, self.nv, self.nw,
                self.v_space, self.w_space
            )
            if changed:
                policy_changed = True
                self.policy[t] = new_policy
        return policy_changed
    
    @staticmethod
    @njit
    def _jit_policy_improvement(costs_t, transitions_t, next_V, current_policy, gamma, nx, ny, nth, nv, nw, v_space, w_space):
        """
        Policy improvement step of the GPI algorithm with Numba JIT compilation.
        """
        new_policy = current_policy.copy()
        policy_changed = False
        for ex_id in prange(nx):
            for ey_id in range(ny):
                for eth_id in range(nth):
                    best_value = np.inf
                    best_v = current_policy[ex_id, ey_id, eth_id, 0]
                    best_w = current_policy[ex_id, ey_id, eth_id, 1]
                    for v_id in range(nv):
                        for w_id in range(nw):
                            cost = costs_t[ex_id, ey_id, eth_id, v_id, w_id]

                            # If the cost is infinite, it means this state-action pair is invalid (e.g., leads to collision), so we skip it.
                            if np.isinf(cost):
                                continue

                            # Get the next state indices from the pre-computed transition matrix
                            next_state = transitions_t[ex_id, ey_id, eth_id, v_id, w_id]
                            nex_id, ney_id, neth_id = next_state[0], next_state[1], next_state[2]
                            expected_value = next_V[nex_id, ney_id, neth_id]
                            value = cost + gamma * expected_value
                            if value < best_value:
                                best_value = value
                                best_v = v_space[v_id]
                                best_w = w_space[w_id]
                    
                    # Check if the policy has changed for this state
                    if (current_policy[ex_id, ey_id, eth_id, 0] != best_v or \
                        current_policy[ex_id, ey_id, eth_id, 1] != best_w):
                        policy_changed = True

                    # Update the policy for this state
                    new_policy[ex_id, ey_id, eth_id, 0] = best_v
                    new_policy[ex_id, ey_id, eth_id, 1] = best_w
        return new_policy, policy_changed

    @utils.timer
    def policy_evaluation(self, num_evals: int):
        """
        Policy evaluation step of the GPI algorithm.
        """
        for _ in utils.tqdm(range(num_evals)):
            for t in range(self.config.T - 1, -1, -1):
                next_t = (t + 1) % self.config.T  # link to the next phase for periodic infinite horizon
                self.V.value[t] = self._jit_policy_evaluation(
                    self.stage_costs[t], 
                    self.transition_matrix[t], 
                    self.V.value[next_t],
                    self.policy[t],
                    self.config.gamma,
                    self.nx, self.ny, self.nth, self.nv, self.nw,
                    self.v_space, self.w_space
                )

    @staticmethod
    @njit
    def _jit_policy_evaluation(costs_t, transition_t, next_V, policy, gamma, nx, ny, nth, nv, nw, v_space, w_space):
        """
        Policy evaluation step of the GPI algorithm.
        """
        V_new = np.full((nx, ny, nth), np.inf)
        
        for ex_id in prange(nx):
            for ey_id in range(ny):
                for eth_id in range(nth):
                    v = policy[ex_id, ey_id, eth_id, 0]
                    w = policy[ex_id, ey_id, eth_id, 1]
                    # Get the corresponding control indices for the given control values in the policy
                    v_id = np.argmin(np.abs(v_space - v))
                    w_id = np.argmin(np.abs(w_space - w))

                    cost = costs_t[ex_id, ey_id, eth_id, v_id, w_id]

                    # If the cost is infinite, it means this state-action pair is invalid (e.g., leads to collision), so we skip it.
                    if np.isinf(cost):
                        continue

                    # Get the next state indices from the pre-computed transition matrix
                    next_state = transition_t[ex_id, ey_id, eth_id, v_id, w_id]
                    nex_id, ney_id, neth_id = next_state[0], next_state[1], next_state[2]
                    expected_value = next_V[nex_id, ney_id, neth_id]
                    V_new[ex_id, ey_id, eth_id] = cost + gamma * expected_value
        return V_new

    @utils.timer
    def compute_policy(self, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the policy for a given number of iterations.
        Args:
            num_iters (int): number of iterations
        """
        # Initialize value function and policy
        self.init_value_function()
        self.init_policy()
        for iter in range(num_iters):
            print(f"GPI Iteration {iter + 1}/{num_iters}")
            policy_changed = self.policy_improvement()
            self.policy_evaluation(num_evals=self.config.num_evals)

            # Check for convergence (if policy does not change, we can stop early)
            if not policy_changed and iter > 0:
                print("Policy converged, stopping early.")
                break

        # Save the final policy and value function to output dir
        np.save(self.policy_dir.as_posix(), self.policy)
        np.save(self.value_dir.as_posix(), self.V.value)

        return self.policy, self.V.value

    @utils.timer
    def load_policy(self):
        """
        Load the policy from output dir.
        """
        return np.load(self.policy_dir) if self.policy_dir.exists() else None
    
    @utils.timer
    def load_value_function(self):
        """
        Load the value function from output dir.
        """
        return np.load(self.value_dir) if self.value_dir.exists() else None
    
    @utils.timer
    def load_mask(self):
        """
        Load the valid error states mask from output dir.
        """
        return np.load(self.mask_dir) if self.mask_dir.exists() else None
    
    @utils.timer
    def load_transition_matrix(self):
        """
        Load the transition matrix from output dir.
        """
        return np.load(self.transition_dir) if self.transition_dir.exists() else None

    @utils.timer
    def load_stage_costs(self):
        """
        Load the stage costs from output dir.
        """
        return np.load(self.stage_cost_dir) if self.stage_cost_dir.exists() else None

    @staticmethod
    def _find_closest_index(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Find the closest index in grid for given values.
        Args:
            grid: [N, ] array of grid points
            values: [M, ] array of values to find closest indices for
        Returns:
            [M, ] array of indices of closest grid points to values
        """
        ids = np.searchsorted(grid, values)
        ids = np.clip(ids, 1, len(grid) - 1)
        
        left = grid[ids - 1]
        right = grid[ids]
        choose_left = np.abs(left - values) < np.abs(right - values)
        return np.where(choose_left, ids - 1, ids)

    @staticmethod
    def _find_closest_index_angle(grid: np.ndarray, angles: np.ndarray) -> np.ndarray:
        # wrap angles to [0, 2*pi)
        ang = np.mod(angles, 2*np.pi)
        grid_mod = np.mod(grid, 2*np.pi)

        # ensure sorted
        order = np.argsort(grid_mod)
        grid_s = grid_mod[order]        # (G,)

        # searchsorted on the circle
        id = np.searchsorted(grid_s, ang)   # (N,)

        # get candidate angles
        G = grid_s.shape[0]
        right = id % G                # wrap id == G to 0
        left  = (id - 1) % G          # wrap -1 to G-1
        a_left  = grid_s[left]         # (N,)
        a_right = grid_s[right]        # (N,)

        # compute wrapped differences for each candidate only
        def ang_diff(a, b):
            d = np.abs(a - b)
            return np.minimum(d, 2*np.pi - d)

        dl = ang_diff(ang, a_left)
        dr = ang_diff(ang, a_right)

        choose_left = dl <= dr
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(G)

        nearest_sorted_id = np.where(choose_left, left, right)
        return inv_order[nearest_sorted_id]
    
    @staticmethod
    def _angle_wrap(angles):
        """
        Wrap the angles to [-pi, pi].
        Args:
            angles: input angles
        Returns:
            wrapped_angles: wrapped angles
        """
        return np.arctan2(np.sin(angles), np.cos(angles))
    
    @staticmethod
    def _car_error_kinematics(cur_err_states: np.ndarray, controls: np.ndarray, cur_ref_state: np.ndarray, next_ref_state: np.ndarray, dt: float):
        """
        Compute the error kinematics given the current error state and control input.
        Args:
            cur_err_states: current error state hypothesiss (N, 3)
            controls: control inputs for the current error state hypothesis (N, 2)
            cur_ref_state: current reference state
            next_ref_state: next reference state
        Returns:
            next_err_states: next error states for the current error state hypothesis (N, 3)
        """
        theta = GPI._angle_wrap(cur_err_states[:, 2] + cur_ref_state[2])
        d_phi = controls[:, 1] * dt / 2  # half step rotation (w*dt/2)
        phi = theta + d_phi

        # use the exact integration of the unicycle model with piecewise constant control over dt.
        N = theta.shape[0]
        sinc_values = np.sinc(d_phi / np.pi)
        rot_3d_z_exact_integration = np.zeros((N, 3, 2))
        rot_3d_z_exact_integration[:, 0, 0] = dt * sinc_values * np.cos(phi)
        rot_3d_z_exact_integration[:, 1, 0] = dt * sinc_values * np.sin(phi)
        rot_3d_z_exact_integration[:, 2, 1] = dt
        F = np.einsum('nij,nj->ni', rot_3d_z_exact_integration, controls)

        ref_err = cur_ref_state - next_ref_state
        next_err_states = cur_err_states + F + ref_err
        next_err_states[:, 2] = GPI._angle_wrap(next_err_states[:, 2])

        return next_err_states
    
    @staticmethod
    def _create_multiband_grid(bands):
        """
        bands: list of (inner, outer, n_points) triples, symmetric around 0.
            For example:
            [ (0.0, 0.1, 21),   # ultra-fine [-0.1, 0.1]
              (0.1, 0.4, 21),   # fine (0.1, 0.4]
              (0.4, 0.8, 11) ]  # coarse (0.4, 0.8]
        """
        grids = []
        for inner, outer, n in bands:
            assert outer > inner >= 0
            # positive side
            g_pos = np.linspace(inner, outer, n, endpoint=False)
            # mirror to negative side
            g = np.concatenate([-g_pos[::-1], g_pos])
            grids.append(g)
        # Always include 0
        grids.append(np.array([0.0]))
        return np.unique(np.concatenate(grids))
