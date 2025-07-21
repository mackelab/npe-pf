# NPE-PF: Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models

This is a minimal repository implementing NPE-PF, a method for simulation-based inference using tabular foundation models, together with its sequential variant TSNPE-PF. See the associated [preprint](https://arxiv.org/abs/2504.17660).

In this implementation, [TabPFNv2](https://github.com/PriorLabs/TabPFN) is used as the tabular foundation model.

**NOTE:** This repository is under active development. In the future, the NPE-PF interface will be aligned with the one used in the [`sbi`](https://github.com/sbi-dev/sbi) package. Expect rough edges and breaking changes.

## Installation

Clone the repository and install with pip:

```bash
git clone https://github.com/mackelab/npe-pf
cd npe-pf
pip install -e .
```

## Usage

```python
import torch
from npe_pf import TabPFN_Based_NPE_PF

prior = ... # torch distribution
simulator = ... # callable
x_o = ... # observation

num_simulations = 1000
thetas = prior.sample((num_simulations,))
xs = simulator(thetas)

posterior_estimator = TabPFN_Based_NPE_PF(prior=prior)
posterior_estimator.append_simulations(thetas, xs)

# NO TRAINING!

num_posterior_samples = 10_000
posterior_samples = posterior_estimator.sample((num_posterior_samples,), x=x_o)
```

See `demo.ipynb` for detailed usage examples of both NPE-PF and TSNPE-PF.

## Testing

Some minimal tests are provided. To run them, make sure you have `pytest` installed. Then use the following command:
```bash
pytest --log-cli-level=INFO tests
```
These tests include timings of the autoregressive and ratio-based log probs.

Some tests should only be run when a GPU is available.
To run fast, CPU-friendly tests, use
```bash
pytest -m fast --log-cli-level=INFO tests
```

## License

Prior Labs License. Built with PriorLabs-TabPFN.

## Citation
```
@article{vetter2025effortless,
  title={Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models},
  author={Vetter, Julius and Gloeckler, Manuel and Gedon, Daniel and Macke, Jakob H},
  journal={arXiv preprint arXiv:2504.17660},
  year={2025}
}
```
