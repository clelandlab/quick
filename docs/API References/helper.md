# Helper

Helper functions and classes, including data-saving, connection, fitting, etc.

> Everything directly exposed to `quick`.

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

## 🔵load_ini

```python
data = quick.load_ini(path)
```

Load the parameters saved in an ini file (in the format produced by `Saver`).

**Parameters**:

- `path` (str) Path to the ini file

**Return**:

- `data` (dict) a Python dictionary

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

The *class* to construct a data saver. The meta information will be saved in a yml file and data points will be saved in a csv file. (An ini file will also be generated for capability of *LabRAD*)

**Parameters**:

- `title` (str) filename (also the title) of the data.
- `path` (str) path to the directory to save the data.
- `indep_params=[]` (list) a list of 2-tuples, specifying meta information for independent variables, in the format of `("Name", "Unit")`
- `dep_params=[]` (list) a list of 2- or 3-tuples, specifying meta information for dependent variables, in the format of `("Name", "Unit")`
- `params={}` (dict) a dictionary of other parameters, will be saved as meta information.

> Most variables and methods are for internal use and therefore not documented here. To save data, use the `write_data` method below.

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
indep_params = [("Frequency", "MHz")]                            # a list of ("Name", "Unit")
dep_params = [("Amplitude", "dB"), ("Data", "", "IQ Amplitude")] # a list of ("Name", "Unit")
params = { "res_ch": 6, "ro_chs": [0], "reps": 1 }
s = quick.Saver("Test Saving", path, indep_params, dep_params, params)
data = [ # data to save is a 2D array
    [1, 100, 200],  # match "Frequency", "Amplitude", "Data" in order
    [1.5, 101, 201] # defined in indep_params and dep_params
]
s.write_data(data)
```

## 🔵evalStr

```python
res = quick.evalStr(s, var)
```

Evaluate a string as f-string with the given local variables.

**Parameters**:

- `s` (str) a given template string
- `var` (dict) given local variables

**Return**:

- `res` (str) evaluated string, treat the given `s` as f-string.

**Example**:

```python
print(quick.evalStr("{k} + 1 = {k + 1}", { "k": 3 }))
# This prints: 3 + 1 = 4
```

## 🔵symmetryCenter

```python
xc = quick.symmetryCenter(x, y, it=3)
```

Very fancy technique to evaluate the most probable symmmetry center for a signal.

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

## 🟢iq2prob

```python
p = quick.iq2prob(Ss, c0, c1)
```

Get the qubit excitation probability (population) from a list of IQ values.

**Parameters**:

- `Ss` (1D ArrayLike[complex]) a list of IQ values in IQ complex plane.
- `c0` (complex) 0-state center in IQ complex plane.
- `c1` (complex) 1-state center in IQ complex plane.

**Return**:

- `p` (float) 1-state probability.

## 🟢iq_rotation

```python
phase_change, r_threshold = quick.iq_rotation(c0, c1)
```

Compute the phase change in the readout pulse and the real threshold, in order to distinguish the 0-state and 1-state.

**Parameters**:

- `c0` (complex) 0-state center in IQ complex plane.
- `c1` (complex) 1-state center in IQ complex plane.

**Return**:

- `phase_change` (float) [deg] phase change to be added on readout pulse.
- `r_threshold` (float) real threshold in I, above which is excited state.

## 🟢iq_scatter

```python
c0, c1, visibility, Fg, Fe, fig = quick.iq_scatter(S0s, S1s, c0=None, c1=None)
```

Compute the center for 0-state and 1-state from measured calibartion data. Compute the visibility and readout fidelity. Plot the IQ scatter and histogram.

**Parameters**:

- `S0s` (1D ArrayLike[complex]) a list of IQ values in IQ complex plane, as the Qubit is in 0 state.
- `S1s` (1D ArrayLike[complex]) a list of IQ values in IQ complex plane, as the Qubit is in 1 state.
- `c0=None` (complex) 0-state center in IQ complex plane. Can be determined if not provided.
- `c1=None` (complex) 1-state center in IQ complex plane. Can be determined if not provided.

**Return**:

- `c0` (complex) 0-state center in IQ complex plane.
- `c1` (complex) 1-state center in IQ complex plane.
- `visibility` (float) computed readout visibility
- `Fg` (float) computed readout fidelity for ground state (0-state)
- `Fe` (float) computed readout fidelity for excited state (1-state)
- `fig` (matplotlib.figure) plotted IQ scatter and histogram.

## 🟢fitT1

```python
popt, perr, rchi2, fig = quick.fitT1(T, S)
```

Fit and plot the Qubit T1 from data.

**Parameters**:

- `T` (1D ArrayLike) a list of pulse delay time.
- `S` (1D ArrayLike) a list of corresponding qubit population.

**Return**:

- `popt` (np.Array(3)) Fitted parameter values. `popt[1]` is the value for T1.
- `perr` (np.Array(3)) Fitted parameter errors. `perr[1]` is the error for T1.
- `rchi2` (float) relative chi square of the fitting.
- `fig` (matplotlib.figure) plotted T1 decay.

**Example**:

```python
data = quick.load_data("path/to/your/data.csv").T
popt, perr, rchi2, fig = quick.fitT1(data[0], data[1])
```

## 🟢fitT2

```python
popt, perr, rchi2, fig = quick.fitT2(T, S, omega=2*np.pi)
```

Fit and plot the Qubit T2 from data.

**Parameters**:

- `T` (1D ArrayLike) a list of pulse delay time.
- `S` (1D ArrayLike) a list of corresponding qubit population.
- `omega=2*np.pi` (float) initial value for angular frequency.

**Return**:

- `popt` (np.Array(4)) Fitted parameter values. `popt[1]` is the value for T2.
- `perr` (np.Array(4)) Fitted parameter errors. `perr[1]` is the error for T2.
- `rchi2` (float) relative chi square of the fitting.
- `fig` (matplotlib.figure) plotted T2 decay.

**Example**:

```python
data = quick.load_data("path/to/your/data.csv").T
omega = quick.estimateOmega(data[0], data[1])
popt, perr, rchi2, fig = quick.fitT2(data[0], data[1], omega=omega)
```

## 🟢fitResonator

```python
p, fig = quick.fitResonator(F, S, fit="circle", p0=[None, None, None, None, None, None, None])
```

Fit and plot the resonator spectroscopy from data. 100 data points away from the resonator dip are required to calculate the `electronic_delay`.

**Parameters**:

- `F` (1D ArrayLike) a list of frequency
- `S` (1D ArrayLike complex) a list of corresponding complex S21.
- `fit="circle` (str) "circle", "amp" or "angle". the target of the fitting.
- `p0=[None, None, None, None, None, None, None]` (1D ArrayLike) initial value of the fitting parameters. If None, then default value will be used. 

**Return**:

- `p` (np.Array(4)) Fitted parameter values in the order of `[Qi, Qc, fr, phi, electronic_delay, background, phase_shift]`
- `fig` (matplotlib.figure) plotted fitting.

**Example**:

```python
# take data in lin mag.
data = quick.load_data("path/to/your/data1.csv", "path/to/your/data2.csv").T
I = data[1] * np.cos(data[2])
Q = data[1] * np.sin(data[2])
S = I + 1j * Q
p, fig = quick.fitResonator(data[0], S, fit="circle")
```

