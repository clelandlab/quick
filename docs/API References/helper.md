# Helper

Helper functions and classes, including data-saving, connection, fitting, etc.

> Everything directly exported to `quick`.

## 🟢connect

```python
soccfg, soc = quick.connect(ip, port=8888, proxy_name="qick")
```

Connect to a QICK board.

**Parameters**:

- `ip` (str) IP address of the QICK board
- `port=8888` (int) Port of the QICK board
- `proxy_name="qick"` (str) Proxy server name of the QICK board

**Return**:

- `soccfg` QICK board socket config
- `soc` QICK board socket

## 🟡getSoc

```python
soccfg, soc = quick.getSoc()
```

Get the socket objects for the last connected QICK board. Mostly for internal use.

**Return**:

- `soccfg` QICK board socket config
- `soc` QICK board socket

## 🟢print_yaml

```python
quick.print_yaml(data)
```

Print a Python object in yaml format.

**Parameters**:

- `data` a Python object, such as dict.

## 🟢load_yaml

```python
data = quick.load_yaml(path)
```

Load a yaml file.

**Parameters**:

- `path` (str) Path to the yaml file

**Return**:

- `data` a Python object

## 🟢save_yaml

```python
content = quick.save_yaml(path, data)
```

Save a Python object to a yaml file.

**Parameters**:

- `path` (str) Path to the proposed yaml file
- `data` a Python object to be saved

**Return**:

- `content` (str) the saved yaml string

## 🟢load_data

```python
data = quick.load_data(*paths)
```

Load arbitrary number of data files (.csv).

**Parameters**:

- `*paths` (str) path to the data files

**Return**:

- `data` (2D Array) combined data. Data rows from all files will be concatenated.

## 🟢Sweep

```python
s = quick.Sweep(config, sweepConfig, progressBar=True)
```

The *class* to construct an iterable. In each iteration, new dictionary will be generated according to the template dictionary `config` and sweeping list set in `sweepConfig`.

**Parameters**:

- `config` (dict) The template dictionary. It will NOT be modified.
- `sweepConfig` (dict) The sweeping list. See the following example.
- `progressBar=True` (bool) Whether to show progress bar.

**Example**:

```python
config = {
	"a": 0,
	"b": 1,
	"c": 2
}
sweepConfig = {
	"a": np.arange(0, 1, 0.1),
	"b": [0, 1, 3, 8, 9]
}

for cfg in quick.Sweep(config, sweepConfig):
	print(cfg) # dictionary with values of "a" and "b" modified.
```

## 🟢Saver

```python
s = quick.Saver(title, path, indep_params=[], dep_params=[], params={})
```

The *class* to construct a data saver. The meta information will be saved in a yml file and data points will be saved in a csv file.

**Parameters**:

- `title` (str) filename (also the title) of the data.
- `path` (str) path to the directory to save the data.
- `indep_params=[]` (list) a list of 2-tuples, specifying meta information for independent variables, in the format of `("Name", "Unit")`
- `dep_params=[]` (list) a list of 2-tuples, specifying meta information for dependent variables, in the format of `("Name", "Unit")`
- `params={}` (dict) a dictionary of meta information and other parameters.

> Most variables and methods are for internal use and therefore not documented here. To save data, use the `write_data` method below.

### - Saver.write_yml

```python
s.write_yml()
```

Write a yml file, recording the metainformation of the current saver. This will overwrite any existing yml file produced by the same saver. This function will be called during the saver initialization. It is recommended to call this function after all data writings to update the completed time.

### - Saver.write_data

```python
s.write_data(data)
```

Write data to a data saver. The data will be **immediately appended** to the data file. Actual file writing is performed in this method. Numerical data will be saved as scientific notation with 10 digits significant figures, eg. `1.234567890e-3`. This function can be called repetitively to keep appending data.

**Parameters**:

- `data` (2D ArrayLike) a list of data rows. Each row should be a list of numerical data, in the exact order defined in `indep_params` and `dep_params`. See the example below.

**Example**:

```python
path = "path/to/directory/"
indep_params = [("Frequency", "MHz")]               # a list of ("Name", "Unit")
dep_params = [("Amplitude", "dB"), ("Data", "")]    # a list of ("Name", "Unit")
params = { "res_ch": 6, "ro_chs": [0], "reps": 1 }  # any dictionary
s = quick.Saver("Test Saving", path, indep_params, dep_params, params)
data = [ # data to save is a 2D array
    [1, 100, 200],  # match "Frequency", "Amplitude", "Data" in order
    [1.5, 101, 201] # defined in indep_params and dep_params
]
s.write_data(data)
# OPTIONAL: after complete, call s.write_yml()
s.write_yml() # this will update the completed time.
```

## 🔵dB2gain

```python
gain = quick.dB2gain(dB)
```

Convert power from dB unit to QICK board gain. 0 dB is the maximum gain.

**Parameters**:

- `dB` (float) power value in dB unit.

**Returns**:

- `gain` (float) value of gain

## 🟢evalStr

```python
res = quick.evalStr(s, var, _var=None)
```

Evaluate a string as f-string with the given local and global variables. All and only the things within `{}` will be evaluated as Python expression. Everything outside `{}` will not be changed. **The string cannot include any other bracket than those that need to be parsed as Python expressions.**

**Parameters**:

- `s` (str) a given template string
- `var` (dict) given local variables
- `_var=None` (dict) given global variables

**Return**:

- `res` (str) evaluated string, treat the given `s` as f-string.

**Example**:

```python
print(quick.evalStr("{k} + 1 = {k + 1}", { "k": 3 })) # This prints: 3 + 1 = 4

mercator_protocol = """
soft_avg: 100
p0_freq: {r_freq}
p0_length: {r_length}
p0_power: {r_power}
r{rr}_p: 0
r{rr}_length: {r_length / 2} # support any expression
r{rr}_phase: {r_phase + 180}
steps:
- type: pulse
  p: 0
  g: {r}
- type: trigger
  t: {r_offset}
- type: wait_auto
- type: delay_auto
  t: {r_relax}
"""
v = dict(quick.experiment.var) # default variables dictionary
cfg = yaml.safe_load(quick.evalStr(mercator_protocol, v))
```

## 🔵symmetryCenter

```python
xc = quick.symmetryCenter(x, y, it=3)
```

Very fancy technique to evaluate the most probable symmetry center for a signal.

**Parameters**:

- `x` (1D ArrayLike) data of independent variable. Better to be equally spaced.
- `y` (1D ArrayLike) data of dependent variable.
- `it=3` (int) number of iteration.

**Return**:

- `xc` (float) estimated symmetry center. Interpolated, not in `x`.

## 🟢estimateOmega

```python
omega = quick.estimateOmega(x, y)
```

Rough estimate the angular frequency using FFT. Useful for initial parameters in curve fitting.

**Parameters**:

- `x` (1D ArrayLike) data of independent variable. Must be equally spaced.
- `y` (1D ArrayLike) data of dependent variable.

**Return**:

- `omega` (float) estimated angular frequency.

## 🟢iq_scatter

```python
phase, threshold, visibility, Fg, Fe, c0, c1, fig = quick.iq_scatter(S0s, S1s, c0=None, c1=None, plot=True)
```

Compute the center for 0-state and 1-state from measured data. Compute the visibility and readout fidelity. Plot the IQ scatter and histogram.

**Parameters**:

- `S0s` (1D ArrayLike[complex]) a list of IQ values in IQ complex plane, as the Qubit is in 0 state.
- `S1s` (1D ArrayLike[complex]) a list of IQ values in IQ complex plane, as the Qubit is in 1 state.
- `c0=None` (complex) 0-state center in IQ complex plane. Can be determined if not provided.
- `c1=None` (complex) 1-state center in IQ complex plane. Can be determined if not provided.
- `plot=True` (bool) whether to plot the IQ scatter and histogram.

**Return**:

- `phase` (float) [deg] phase change to be added on ADC to get horizontal state-distinguish
- `threshold` (float) threshold in I (after the phase change), above which is excited state
- `visibility` (float) computed readout visibility
- `Fg` (float) computed readout fidelity for ground state (0-state)
- `Fe` (float) computed readout fidelity for excited state (1-state)
- `c0` (complex) 0-state center in IQ complex plane.
- `c1` (complex) 1-state center in IQ complex plane.
- `fig` (matplotlib.figure) plotted IQ scatter and histogram. `None` if `plot=False`.

## 🟢fitT1

```python
p, perr, r2, fig = quick.fitT1(T, S, plot=True)
```

Fit and plot the Qubit T1 from data.

**Parameters**:

- `T` (1D ArrayLike) a list of pulse delay time.
- `S` (1D ArrayLike) a list of corresponding qubit population.
- `plot=True` (bool) whether to plot the fitting.

**Return**:

- `p` (np.Array(3)) Fitted parameter values. `p[1]` is the value for T1.
- `perr` (np.Array(3)) Fitted parameter errors. `perr[1]` is the error for T1.
- `r2` (float) R-squared of the fitting.
- `fig` (matplotlib.figure) fitting plot. `None` if `plot=False`.

## 🟢fitT2

```python
p, perr, r2, fig = quick.fitT2(T, S, omega=2*np.pi, T2=20.0, plot=True)
```

Fit and plot the Qubit T2 from data.

**Parameters**:

- `T` (1D ArrayLike) a list of pulse delay time.
- `S` (1D ArrayLike) a list of corresponding qubit population.
- `omega=2*np.pi` (float) initial value for angular frequency.
- `T2=20.0` (float) initial guess value for T2.
- `plot=True` (bool) whether to plot the fitting.

**Return**:

- `p` (np.Array(4)) Fitted parameter values. `p[1]` is the value for T2.
- `perr` (np.Array(4)) Fitted parameter errors. `perr[1]` is the error for T2.
- `r2` (float) R-squared of the fitting.
- `fig` (matplotlib.figure) fitting plot. `None` if `plot=False`.

**Example**:

```python
data = quick.load_data("path/to/your/data.csv").T
omega = quick.estimateOmega(data[0], data[1])
p, perr, r2, fig = quick.fitT2(data[0], data[1], omega=omega)
```

## 🟢fitResonator

```python
p, perr, r2, fig = quick.fitResonator(F, S, fit="circle", p0=[None, None, None, None], plot=True)
```

Circle fit of inverse S21 for quality factor of resonator.

**Parameters**:

- `F` (1D ArrayLike) a list of frequency
- `S` (1D ArrayLike complex) a list of corresponding complex S21.
- `fit="circle"` (str) "circle", "amp" or "arg". the target of the fitting.
- `p0=[None, None, None, None]` (1D ArrayLike) initial value of the fitting parameters in the order of `[Qi, Qc, fr, phi]`. If `None`, then default value will be used. 
- `plot=True` (bool) whether to plot the fitting.

**Return**:

- `p` (np.Array(4)) Fitted parameter values in the order of `[Qi, Qc, fr, phi]`
- `perr` (np.Array(4)) Fitted parameter errors.
- `r2` (float) R-squared of the fitting.
- `fig` (matplotlib.figure) fitting plot. `None` if `plot=False`.

**Example**:

```python
data = quick.load_data("path/to/data1.csv", "path/to/data2.csv").T # combine two scan
p, perr, r2, fig = quick.fitResonator(data[0], data[3] + 1j * data[4])
```

