# Get Started

QuICK is a universal wrap of [QICK](https://github.com/openquantumhardware/qick).

<div>
  <a style="margin: 0.25rem;" href="https://clelandlab-quick.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/clelandlab-quick?style=for-the-badge&logo=readthedocs&logoColor=white"></a>
  <a style="margin: 0.25rem;" href="https://pypi.org/project/clelandlab-quick/"><img src="https://img.shields.io/pypi/v/clelandlab-quick?style=for-the-badge&logo=pypi&logoColor=white"></a>
  <a style="margin: 0.25rem;" href="https://github.com/clelandlab/quick"><img src="https://img.shields.io/github/stars/clelandlab/quick?style=for-the-badge&logo=github"></a>
</div>

![](./Images/overview.png)

## Installation

> This is the installation on your PC. For QICK Board setup, see [here](./Tutorials/qick).

Install this package with `pip`:

```
pip install clelandlab-quick
```

> It is recommended to use a conda environment.

```python
import quick

# connect to the QICK board
soccfg, soc = quick.connect(QICK_IP)
```

## Document Convention

**All and only *class* are capitalized.**

The APIs fall into several accessibility:

- 🟢 very useful
- 🔵 sometimes useful
- 🟡 mostly internal use
- 🔴 broken or deprecated
