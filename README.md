# ECE276B-PR3-Infinite-Horizon-Stochastic-Optimal-Control
SP 26 ECE 276B Project 3: Infinite-Horizon Stochastic Optimal Control

## Course Overview
This is Project 3 for [ECE 276B: Planning & Learning in Robotics](https://natanaso.github.io/ece276b/) at UCSD, taught by Professor [Nikolay Atanasov](https://natanaso.github.io/).

## Project Description
This project focuses on solving ***infinite-horizon* stochastic optimal control** problems to develop safe trajectory tracking techniques for a ground differential-drive robot.

## Prerequisites
The code is only tested with miniconda environment.
- Miniconda Installed: https://docs.anaconda.com/miniconda/install/#quick-command-line-install
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    ```
    After installing, close and reopen your terminal application or refresh it by running the following command:
    ```bash
    source ~/miniconda3/bin/activate
    conda init --all
    ```
- Use `conda` to create a `python3.11` virtual environment (`ece276b_pr3`), and install required packages:
    ```bash
    conda create -n ece276b_pr3 python=3.11
    conda activate ece276b_pr3
    pip3 install -r requirements.txt
    ```
- Whenever creating a **new terminal session**, do:
    ```bash
    conda deactivate
    conda activate ece276b_pr3
    ```

## Running the Project
### Standard Run:

Run tests using `src/main.py`:
```bash
# TODO
```

## Features
- TODO

## Output Graphs
TODO

### Part 1:
TODO

### Part 2:
TODO

### Part 3:
TODO

## Acknowledgments
This project is part of the **ECE 276B** course at **UC San Diego**, which is inspired by the teaching and research of various professors in the field of robotics and planning.