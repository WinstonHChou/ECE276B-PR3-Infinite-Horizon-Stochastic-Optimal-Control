# ECE276B-PR3-Infinite-Horizon-Stochastic-Optimal-Control
SP 26 ECE 276B Project 3: Infinite-Horizon Stochastic Optimal Control

## Course Overview
This is Project 3 for [ECE 276B: Planning & Learning in Robotics](https://natanaso.github.io/ece276b/) at UCSD, taught by Professor [Nikolay Atanasov](https://natanaso.github.io/).

## Project Overview
This project studies trajectory tracking for a unicycle-like car model in a 2D obstacle environment.

Two controllers are implemented and compared:
- Certainty Equivalent Control (CEC): receding-horizon nonlinear optimization (CasADi + IPOPT)
- Generalized Policy Iteration (GPI): discretized infinite-horizon dynamic programming

The target trajectory is a periodic lemniscate (figure-eight). Dynamics are propagated with exact discrete-time integration, and MuJoCo is used as an additional simulator for validation.

## Repository Structure
- `src/main.py`: entry point, controller selection, simulation loop
- `src/utils.py`: dynamics, constants, trajectory, visualization
- `src/cec.py`: CEC formulation and solver
- `src/gpi.py`: GPI discretization, transition/cost precomputation, policy iteration
- `src/mujoco_car.py`: MuJoCo simulator wrapper
- `src/fig/`: generated figures
- `src/gpi_outputs/`: cached GPI artifacts (`policy.npy`, `value.npy`, transition/cost tensors)
- `report/[ECE 276B] PR 3 Report.tex`: final writeup

## Environment Setup
Tested with Python 3.11 in a conda environment.

```bash
conda create -n ece276b_pr3 python=3.11
conda activate ece276b_pr3
pip install -r requirements.txt
```

For each new terminal session:

```bash
conda activate ece276b_pr3
```

## How to Run
Run from the repository root.

### CEC in numerical simulator
```bash
python src/main.py -c cec
```

### CEC in MuJoCo simulator
```bash
python src/main.py -c cec --mujoco
```

### GPI rollout
```bash
python src/main.py -c gpi
```

### Save animation output
Add `--save` to any run command.

```bash
python src/main.py -c cec --save
```

## Key Parameters
Current default settings (from code):
- `dt = 0.5 s`
- World bounds: `[-3, 3] x [-3, 3]`
- Robot radius: `0.3`
- Velocity limits: `v in [0.1, 1.0]`, `w in [-1.0, 1.0]`
- Process noise sigma: `[0.04, 0.04, 0.004]`
- CEC horizon: `T = 10`
- GPI horizon: `T = 100` (one trajectory period with `dt = 0.5`)
- CEC control weight: `R = diag(0.1, 0.1)`
- GPI control weight: `R = diag(0.5, 0.5)`

## Current Results Summary (dt = 0.5)
Measured from current runs:

| Method | Avg Iteration Time (ms) | Final Translational Error | Final Rotational Error |
|---|---:|---:|---:|
| CEC (Numerical, exact integration) | 69.22 | 55.14 | 43.74 |
| CEC (MuJoCo) | 692.83 | 65.94 | 28.99 |
| GPI (deterministic, no process noise) | 0.37 | 142.52 | 94.99 |

Interpretation:
- CEC currently gives better tracking quality in this project.
- GPI is much faster online but currently less accurate and still incomplete.
- The reported GPI rollout is deterministic (no process noise in the transition model).
- A complete stochastic GPI benchmark should incorporate stochastic transition backups, which is left as future work.

## Figures
Generated example figures:
- `src/fig/cec_dt_0.5.png`
- `src/fig/cec_mujoco_dt_0.5.png`
- `src/fig/gpi_dt_0.5_no_noise.png`

## Notes
- GPI precomputation can be expensive the first time because transition and stage-cost tensors are cached to `src/gpi_outputs/`.
- If you change discretization sizes or horizon settings, clear/update cached outputs accordingly.

## Report
The full writeup is in:
- `report/[ECE 276B] PR 3 Report.tex`

## Acknowledgment
This project is part of ECE 276B at UC San Diego.