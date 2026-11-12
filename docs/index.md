# Get Started

QuICK provides experiment routines and pulse protocols for [QICK](https://github.com/openquantumhardware/qick) firmware.

<div>
  <a style="margin: 0.25rem;" href="https://clelandlab-quick.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/clelandlab-quick?style=for-the-badge&logo=readthedocs&logoColor=white"></a>
  <a style="margin: 0.25rem;" href="https://pypi.org/project/clelandlab-quick/"><img src="https://img.shields.io/pypi/v/clelandlab-quick?style=for-the-badge&logo=pypi&logoColor=white"></a>
  <a style="margin: 0.25rem;" href="https://github.com/clelandlab/quick"><img src="https://img.shields.io/github/stars/clelandlab/quick?style=for-the-badge&logo=github"></a>
</div>

![](./Images/overview.png)

## Installation

Use Python 3.13+ on the PC and install this checkout from its repository root. An independently published release may not contain the native changes. Install the separate `board_setup_next` package on the board before connecting; see [Board Setup](Tutorials/qick.md).

Install this package with `pip`:

```
pip install clelandlab-quick
```

> It is recommended to use a conda environment.

```python
import quick

# connect to the native board server on TCP 9999
soccfg, soc = quick.connect(QICK_IP)
```

## Document Convention

**All and only *class* are capitalized.**

The APIs fall into several accessibility:

- 🟢 very useful
- 🔵 sometimes useful
- 🟡 mostly internal use
- 🔴 broken or deprecated
