# QuICK

QuICK is a useful wrap of [QICK](https://github.com/openquantumhardware/qick).

<div>
  <a style="margin: 0.25rem;" href="https://clelandlab-quick.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/clelandlab-quick?style=for-the-badge&logo=readthedocs&logoColor=white"></a>
  <a style="margin: 0.25rem;" href="https://pypi.org/project/clelandlab-quick/"><img src="https://img.shields.io/pypi/v/clelandlab-quick?style=for-the-badge&logo=pypi&logoColor=white"></a>
  <a style="margin: 0.25rem;" href="https://github.com/clelandlab/quick"><img src="https://img.shields.io/github/stars/clelandlab/quick?style=for-the-badge&logo=github"></a>
</div>

## Installation

> This is the installation on your PC. For QICK Board setup, see [here](https://clelandlab-quick.readthedocs.io/en/latest/Tutorials/qick).

Install this package with `pip`:

```
pip install clelandlab-quick
```

> It is recommended to use a conda environment.

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
