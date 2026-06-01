from time import time
import numpy as np
from utils import *
# from cec import CEC
from mujoco_car import MujocoCarSim
import argparse


CONTROLLER_LOOKUP = {
    SimpleController.name: SimpleController,
    # CEC.name: CEC,
}

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--controller", choices=CONTROLLER_LOOKUP.keys(), help="Controller to use", required=True)
parser.add_argument("--save", dest="save", action="store_true", help="Save the simulation as a gif")
parser.add_argument("--mujoco", action="store_true", help="Use MuJoCo simulation")
args = parser.parse_args()
USE_MUJOCO = args.mujoco
SAVE_GIF = args.save


def main():
    # Obstacles in the environment (x, y, radius)
    obstacles = np.array([
        [2.35, 0.95, 0.5],
        [-2.35, -0.95, 0.5],
        [1.0, 0.0, 0.5],
        [-1.0, 0.0, 0.5],
    ])

    # Params
    traj = lemniscate
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []

    # Start main loop
    main_loop = time()  # return time in sec

    # Initialize state
    cur_state = np.array([X_INIT, Y_INIT, THETA_INIT])
    cur_iter = 0

    # Initialize MuJoCo simulation environment
    mujoco_sim = None
    if USE_MUJOCO:
        mujoco_sim = MujocoCarSim()

    # Initialize controller
    controller_cls = CONTROLLER_LOOKUP[args.controller]
    controller = controller_cls(time_step=TIME_STEP)

    # Main loop
    while cur_iter * TIME_STEP < SIM_TIME:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * TIME_STEP
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
            next_state = car_next_state(TIME_STEP, cur_state, control, noise=True)

        # Update current state
        cur_state = next_state

        # Loop time
        t2 = time()
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
    visualize(car_states, ref_traj, obstacles, times, TIME_STEP, save=SAVE_GIF)


if __name__ == "__main__":
    main()
