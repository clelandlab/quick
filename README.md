# QuICK

QuICK is a universal wrap of [QICK](https://github.com/openquantumhardware/qick).

<div style="display: flex; flex-direction: row; flex-wrap: wrap; justify-content: center; align-items: center;">
  <a style="margin: 0.25rem; display: block;" href="https://clelandlab-quick.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/clelandlab-quick?style=for-the-badge&logo=readthedocs&logoColor=white"></a>
  <a style="margin: 0.25rem; display: block;" href="https://pypi.org/project/clelandlab-quick/"><img src="https://img.shields.io/pypi/v/clelandlab-quick?style=for-the-badge&logo=pypi&logoColor=white"></a>
  <a style="margin: 0.25rem; display: block;" href="https://github.com/clelandlab/quick"><img src="https://img.shields.io/github/stars/clelandlab/quick?style=for-the-badge&logo=github"></a>
</div>

## Installation

> This is the installation on your PC. For QICK Board setup, see [here](https://clelandlab-quick.readthedocs.io/en/latest/Tutorials/qick).

Install this package with `pip`:

```
pip install clelandlab-quick
```

Then you can import it in your Python code:

```python
import quick
```

## Layers

QuICK has several layers of complexity.

- `quick.auto` Automation of Qubit Measurements
- `quick.experiment` Experiment Routines for Qubit Measurements
- `quick.Mercator` Mercator Protocol for Pulse Sequence Program
- `qick` the QICK firmware

![](https://clelandlab-quick.readthedocs.io/en/latest/Images/overview.png)

## Cite this work

```bibtex
@article{li2026large,
  title={Large Language Model-Assisted Superconducting Qubit Experiments},
  author={Li, Shiheng and Miller, Jacob M and Lee, Phoebe J and Andersson, Gustav and Conner, Christopher R and Joshi, Yash J and Karimi, Bayan and King, Amber M and Malc, Howard L and Mishra, Harsh and others},
  journal={arXiv preprint arXiv:2603.08801},
  year={2026}
}
```
