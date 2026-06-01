from time import time
import numpy as np
import utils
# from cec import CEC
from mujoco_car import MujocoCarSim
import argparse


CONTROLLER_LOOKUP = {
    utils.SimpleController.name: utils.SimpleController,
    # CEC.name: CEC,
}

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--controller", choices=CONTROLLER_LOOKUP.keys(), help="Controller to use", required=True)
parser.add_argument("--mujoco", action="store_true", help="Use MuJoCo simulation")
args = parser.parse_args()
USE_MUJOCO = args.mujoco


def main():
    # Obstacles in the environment (x, y, radius)
    obstacles = np.array([
        [2.35, 0.95, 0.5],
        [-2.35, -0.95, 0.5],
        [1.0, 0.0, 0.5],
        [-1.0, 0.0, 0.5],
    ])

    # Params
    traj = utils.lemniscate
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []

    # Start main loop
    main_loop = time()  # return time in sec

    # Initialize state
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0

    # Initialize MuJoCo simulation environment
    mujoco_sim = None
    if USE_MUJOCO:
        mujoco_sim = MujocoCarSim()

    # Initialize controller
    controller_cls = CONTROLLER_LOOKUP[args.controller]
    controller = controller_cls()

    # Main loop
    while cur_iter * utils.time_step < utils.sim_time:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        control = controller(cur_iter, cur_state, cur_ref)
        ################################################################

        # Apply control input
        if USE_MUJOCO:
            next_state = mujoco_sim.car_next_state(control)
        else:
            next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)

        # Update current state
        cur_state = next_state

        # Loop time
        t2 = utils.time()
        compute_time = t2 - t1
        times.append(compute_time)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])

        # Debug print
        print(f"Iter {cur_iter}, time {cur_time:.2f}s, compute time {compute_time:.4f}s")
        print(f"Current state: {cur_state}, Reference state: {cur_ref}")
        print(f"Current control: {control}")
        print(f"Current error state: {cur_err}")
        print(f"Cumulative Error - Trans: {error_trans}, Rot: {error_rot}")
        print("======================")

        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final error_trains: ", error_trans)
    print("Final error_rot: ", error_rot)

    # Proper shut down of MuJoCo
    if USE_MUJOCO:
        mujoco_sim.viewer_handle.close()

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=True)


if __name__ == "__main__":
    main()
