# ReEKF-SORT

The current ReEKF-SORT documentation, final configuration, reproduction
commands, and verified results are maintained in [README.md](README.md).

The paper-final configuration uses:

- `confidence_cost_mode=absolute`
- `lambda_conf=1.2`
- `virtual_update_interval=6`
- `virtual_obs_noise_scale=10.0`
- `use_angle_cost=false`
- `lambda_angle=0.0`