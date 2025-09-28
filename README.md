# `PyRoki`: Python Robot Kinematics Library

**[Project page](https://pyroki-toolkit.github.io/) &bull;
[arXiv](https://arxiv.org/abs/2505.03728)**

`PyRoki` is a modular, extensible, and cross-platform toolkit for kinematic optimization, all in Python.

Core features include:

- Differentiable robot forward kinematics model from a URDF.
- Automatic generation of robot collision primitives (e.g., capsules).
- Differentiable collision bodies with numpy broadcasting logic.
- Common cost implementations (e.g., end effector pose, self/world-collision, manipulability).
- Arbitrary costs, autodiff or analytical Jacobians.
- Integration with a [Levenberg-Marquardt Solver](https://github.com/brentyi/jaxls) that supports optimization on manifolds (e.g., [lie groups](https://github.com/brentyi/jaxlie))
- Cross-platform support (CPU, GPU, TPU) via JAX.

Please refer to the [documentation](https://chungmin99.github.io/pyroki/) for more details, features, and usage examples.

---

## Installation

You can install `pyroki` with `pip`, on Python 3.10+:

```
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```

## Finding the most robust sit base pose range

1. Collect sit terminal states and trajectories in [`humanoid_eetrack` with branch `feature/sitting/collect_terminal_states`](https://github.com/LibertyRoboticsInc/humanoid_eetrack/tree/feature/sitting/collect_terminal_states)
    * Make `logs/rsl_rl/g1_eetrack/<SITTING_POLICY_EXP_PATH>` directory. E.g. `logs/rsl_rl/g1_eetrack/sitting_right_arm`
    * Put sit policy weight file under the `logs/rsl_rl/g1_eetrack/<SITTING_POLICY_EXP_PATH>`. E.g. `logs/rsl_rl/g1_eetrack/sitting_right_arm/model_15050.pt`
    * This will produce sit terminal states buffer under the same directory of sit weight file. E.g. `logs/rsl_rl/g1_eetrack/sitting_right_arm/terminal_states_11538.pt`

```bash
./do_collect.sh --load_run <SITTING_POLICY_EXP_PATH> --headless --num_envs 4096 --num_states 10000
# E.g.
./do_collect.sh --load_run sitting_right_arm --headless --num_envs 4096 --num_states 10000
```

2. Put the sit terminal states (e.g. terminal_states_11538.pt) under the examples/eetrack directory. E.g. examples/eetrack/terminal_states_11538.pt

3. Convert `.pt` file to `.npz` file. This will print the npz file path. E.g. eetrack/terminal_states_11538.npz
```bash
cd examples
python3 utils/sit_terminal_states_pt_to_npz.py <SIT_TERMINAL_STATES_FILE>
# E.g.
python3 utils/sit_terminal_states_pt_to_npz.py eetrack/terminal_states_11538.pt
```

4. Change `sit_terminal_states_path` in `examples/eetrack/config.yaml` to the saved npz file from 3. E.g. `sit_terminal_states_path: eetrack/terminal_states_11538.npz`

5. Perform base pose sampling and evaulation through trajectory optimization foe each sit target height and welding object z height.
    * To just know code works well, reduce `n_samples` in `examples/eetrack/config.yaml`.
    * Note: the best `sit_target_height` from the last sitting policy is `0.43`

```bash
cd examples
# In terminal 1,
CUDA_VISIBLE_DEVICES=0 python3 run_batch_pipeline_sweep.py --sit_target_height 0.4 --z_height 0.3
# In terminal 2, change sit_target_height.
CUDA_VISIBLE_DEVICES=1 python3 run_batch_pipeline_sweep.py --sit_target_height 0.41 --z_height 0.3
# In terminal 3, also change sit_target_height.
CUDA_VISIBLE_DEVICES=2 python3 run_batch_pipeline_sweep.py --sit_target_height 0.42 --z_height 0.3
# ...
# Do this as much as you want to change sit_target_height.
```

6. Check collision during sitting for the success base pose samples.
    * Note: This will produce new exp folder with name `sit_coll_filtered_exp` under `files/batch_pipeline_h40_z300`
```bash
cd examples
python3 17_check_sitting_collision.py --sit_target_height 0.4 --z_height 0.3
python3 17_check_sitting_collision.py --sit_target_height 0.41 --z_height 0.3
python3 17_check_sitting_collision.py --sit_target_height 0.42 --z_height 0.3
# ...
```

7. Run CMA-ES to find the most wide sit base pose range. Change `exp_prefix` to `batch_pipeline_h<SIT_TARGET_HEIGHT_CM>_z<Z_HEIGHT_MM>`. E.g. `batch_pipeline_h40_z300`.
    * Note: Only check the result from `sit_coll_filtered_exp` not `dummy_exp`.

```bash
cd examples
python utils/cma_es.py --log_dir files --exp_prefix batch_pipeline_h40_z300 --algo cmaes --save_cmaes_result --animate
python utils/cma_es.py --log_dir files --exp_prefix batch_pipeline_h41_z300 --algo cmaes --save_cmaes_result --animate
python utils/cma_es.py --log_dir files --exp_prefix batch_pipeline_h42_z300 --algo cmaes --save_cmaes_result --animate
# ...
```

8. Compare printed sit pose range and success rate from above code and select the best sit pose.

## Status

_May 6, 2025_: Initial release

We are preparing and will release by _May 16, 2025_:

- [x] Examples + documentation for hand / humanoid motion retargeting
- [x] Documentation for using manually defined Jacobians
- [x] Support with Python 3.10+

## Limitations

- **Soft constraints only**: We use a nonlinear least-squares formulation and model joint limits, collision avoidance, etc. as soft penalties with high weights rather than hard constraints.
- **Static shapes & JIT overhead**: JAX JIT compilation is triggered on first run and when input shapes change (e.g., number of targets, obstacles). Arrays can be pre-padded to vectorize over inputs with different shapes.
- **No sampling-based planners**: We don't include sampling-based planners (e.g., graphs, trees).
- **Collision performance**: Speed and accuracy comparisons against other robot toolkits such as CuRobo have not been extensively performed, and is likely slower than other toolkits for collision-heavy scenarios.

The following are current implementation limitations that could potentially be addressed in future versions:

- **Joint types**: We only support revolute, continuous, prismatic, and fixed joints. Other URDF joint types are treated as fixed joints.
- **Collision geometry**: We are limited to sphere, capsule, halfspace, and heightmap geometries. Mesh collision is approximated as capsules.
- **Kinematic structures**: We only support kinematic trees; no closed-loop mechanisms or parallel manipulators.

## Citation

This codebase is released with the following preprint.

<table><tr><td>
    Chung Min Kim*, Brent Yi*, Hongsuk Choi, Yi Ma, Ken Goldberg, Angjoo Kanazawa.
    <strong>PyRoki: A Modular Toolkit for Robot Kinematic Optimization</strong>
    arXiV, 2025.
</td></tr>
</table>

<sup>\*</sup><em>Equal Contribution</em>, <em>UC Berkeley</em>.

Please cite PyRoki if you find this work useful for your research:

```
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

Thanks!
