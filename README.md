# beyond_golden_ratio

Supplementary code (in Julia 1.7) for the [paper](https://arxiv.org/abs/2212.13955):
```
@article{alacaoglu2022beyond,
  title={Beyond the Golden Ratio for Variational Inequality Algorithms},
  author={Alacaoglu, Ahmet and B{\"o}hm, Axel and Malitsky, Yura},
  journal={arXiv preprint arXiv:2212.13955},
  year={2022}
}
```

Collects popular algorithms for variational inequalities + test problems. Some of them monotone. Some of them weak Minty.

## Usage

The `problems/` folder contains all problem instances.

All algorithms can be found in `utils/methods.jl`.

To reproduce the experiments from the paper, simply run `run_files/plots_for_graal_paper.jl`.
