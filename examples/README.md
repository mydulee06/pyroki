# Welding Robot Optimization Pipeline with PyRoki

This repository provides an end-to-end pipeline for finding the optimal base position for a G1 humanoid robot to perform welding tasks, built on top of the Python robot kinematics library, PyRoki.

## 🚀 Key Feature: eetrack Welding Optimization Pipeline

The core feature of this repository is finding the optimal base (pelvis) position for the G1 robot to successfully execute a given welding path. The entire pipeline can be run at once using the `0_run_batch_pipeline.py` script, which internally follows these steps:

1. **Candidate Generation (`14_batch_eetrack.py`)**: Generates numerous candidates by sampling various positions of the welding object within a defined space and optimizing the robot arm's joint trajectory for each position. The results are saved as `.json` files.

2. **Inverse Transform (`utils/inverse_transform.py`)**: Inversely calculates the relative position of the robot base (relative to the mid-sole-link) from the joint trajectory of each generated candidate.

3. **Data Conversion (`utils/json_to_pt_converter.py`)**: Converts the inverse-transformed results (JSON) into PyTorch tensor (`.pt`) files for the next step's input.

4. **Optimal Solution Search (`utils/cma_es.py`)**: Uses the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm to find a single optimal solution with the highest success rate among the numerous candidate base positions.

### How to Run the Full Pipeline

You can run the entire pipeline with the command below. The robot's sitting height (`sit_target_height`) is a crucial parameter and can be adjusted as needed.

```bash
# Run the full pipeline (with a sitting height of 0.37m)
python examples/0_run_batch_pipeline.py --sit_target_height 0.37

# If you have already generated candidates (.json), skip the first step and run
python examples/0_run_batch_pipeline.py --sit_target_height 0.37 --skip_batch_eetrack
```

### Running Pipeline Steps Individually

While `0_run_batch_pipeline.py` is recommended for most cases, you can also run each step of the pipeline individually. This is useful for debugging or if you only need to re-run a specific part of the process. The `--sit_target_height` argument is used across all scripts to specify the directory for inputs and outputs.

#### Step 1: Generate Candidates (`14_batch_eetrack.py`)

This script samples welding object poses and generates joint trajectory candidates for each.

```bash
python examples/14_batch_eetrack.py --num_samples 1000 --use_sit_pose true --sit_target_height 0.37
```

- `--num_samples`: The number of candidate poses to generate.
- `--use_sit_pose`: Specifies whether to use the robot's sitting posture.
- `--sit_target_height`: Defines the target height for the sitting pose and the output directory.

#### Step 2: Inverse Transform (`utils/inverse_transform.py`)

Calculates the robot's base position from the generated joint trajectories.

```bash
python examples/utils/inverse_transform.py --sit_target_height 0.37
```

- This script reads the `.json` files generated in Step 1 from the directory corresponding to the `sit_target_height`.

#### Step 3: Convert Data to Tensors (`utils/json_to_pt_converter.py`)

Converts the JSON output from the previous step into PyTorch `.pt` files for efficient processing.

```bash
python examples/utils/json_to_pt_converter.py --sit_target_height 0.37
```

- Reads the output from Step 2 and saves it as a `.pt` file in the same directory structure.

#### Step 4: Find Optimal Solution with CMA-ES (`utils/cma_es.py`)

Runs the CMA-ES algorithm on the tensor data to find the single best base position.

```bash
python examples/utils/cma_es.py --sit_target_height 0.37
```

- Uses the `.pt` file from Step 3 to perform the final optimization.

## 🛠️ Key Files and Directories for eetrack

- `examples/0_run_batch_pipeline.py`: The **main execution file**.
- `examples/13_eetrack.py`: A script to visually check the End-Effector tracking optimization for a single welding path.
- `examples/14_batch_eetrack.py`: Processes tracking optimization for various welding object positions in a batch to generate candidates.
- `examples/15_mid_sole_sampling_vis.py`: A script for sampling robot base positions for a specific welding task and visualizing the optimization results for testing.
- `examples/16_batch_mid_sole_sampling.py`: The batch processing version of `15_mid_sole_sampling_vis.py`.
- `examples/eetrack/`: A directory containing eetrack-related configurations, robot models, and welding object data.
  - `config.yaml`: A configuration file that defines all major pipeline parameters, such as optimization weights, robot settings, and collision pairs.
  - `welding_task_config.yaml`: A file that defines task-related settings, such as the target position for the welding task.
  - `robots/`: Contains the URDF model files for the G1 robot.
  - `weld_objects/`: Contains the data for the welding objects used to generate welding paths.

## ⚙️ Installation and Setup

In a Python 3.10 or higher environment, install the necessary libraries with the following commands:

```bash
git clone https://github.com/mydulee06/pyroki.git
cd pyroki
pip install -e .
```

## PyRoki: Python Robot Kinematics Library

PyRoki is a modular, extensible, and cross-platform kinematics optimization toolkit written in Python.

Core features include:

- Differentiable robot forward kinematics model generation from URDF.
- Automatic generation of robot collision primitives (e.g., capsules).
- Differentiable collision objects using NumPy broadcasting logic.
- Implementation of common cost functions like End-Effector pose, self/external collision, and manipulability.
- Support for defining arbitrary cost functions via automatic differentiation or analytical Jacobians.
- Integration with various optimization solvers (e.g., Levenberg-Marquardt).
- Cross-platform support for CPU, GPU, and TPU through JAX.

For more detailed information, please refer to the [official documentation](https://chungmin99.github.io/pyroki/).

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If PyRoki was helpful for your research, please cite it as follows:

```bibtex
@misc{pyroki2025,
  title={PyRoki: A Modular Toolkit for Robot Kinematic Optimization},
  author={Chung Min Kim* and Brent Yi* and Hongsuk Choi and Yi Ma and Ken Goldberg and Angjoo Kanazawa},
  year={2025},
  eprint={2505.03728},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2505.03728},
}
```