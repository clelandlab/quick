# Auto

Automatic scripts for qubit measurements.

> Everything exposed to `quick.auto`.

## 🔵BaseAuto

```python
a = quick.auto.BaseAuto(var, silent=False, data_path=None, soccfg=None, soc=None)
```

General base *class* for other auto scripts.

**Parameters**:

- `var` experimental variables. It will NOT be modified.
- `silent=False` (bool) whether to avoid any printing.
- `data_path=None` (str) directory to save data. If not provided, data will not be saved to file.
- `soccfg=None` QICK board socket config object. If not provided, the last connected one will be used by calling `quick.getSoc()`.
- `soc=None` QICK board socket object.

### - BaseAuto.var

```python
a.var
```

The experiment variable dictionary used in the experiment.

### - BaseAuto.data

```python
a.data
```

Data measured or loaded. Will be transposed from the raw data.

The Mercator instance created by the experiment.

### - BaseAuto.load_data

```python
a.load_data(*paths)
```

Load and transpose data using `quick.load_data`.

**Parameters**:

- `*paths` (str) arbitrary number of data paths can be passed in. Multiple data will be combined.

### - BaseAuto.update

```python
a.update(v)
```

Update relevant variables in external dictionary.

**Parameters**:

- `v` (dict) experiment variable to be updated. **Will be modified!**

## 🟢Resonator

> Base *class*: `BaseAuto`

```python
a = quick.auto.Resonator(**kwargs)
```

Determine the readout power and readout frequency from PowerSpectroscopy.

### - Resonator.calibrate

```python
var, fig = a.calibrate(**kwargs)
```

Calibrate the experiment variables from data. Use `quick.experiment` to acquire data when data is not available.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢QubitFreq

> Base *class*: `BaseAuto`

```python
a = quick.auto.QubitFreq(**kwargs)
```

Determine the qubit frequency from QubitSpectroscopy.

### - QubitFreq.calibrate

```python
var, fig = a.calibrate(q_freq_min=3000, q_freq_max=5000, q_gain=0.5, **kwargs)
```

Run the calibration.

**Parameters**:

- `q_freq_min=3000` (float) [MHz] mininum qubit frequency.
- `q_freq_max=5000` (float) [MHz] maximum qubit frequency.
- `q_gain=0.5` (float) [0, 1] initial gain for qubit pulse.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢PiPulseLength

> Base *class*: `BaseAuto`

```python
a = quick.auto.PiPulseLength(**kwargs)
```

Determine the qubit pulse length from Rabi, sweeping pi pulse length and fit the result.

### - PiPulseLength.calibrate

```python
var, fig = a.calibrate(q_length_max=0.5, cycles=[], tol=0.5, **kwargs)
```

Run the calibration.

**Parameters**:

- `q_length_max=0.5` (float) [us] maximum qubit pulse length.
- `cycles=[]` (list) extra cycles in Rabi. Cycle 0 (one pi pulse) will always be included.
- `tol=0.5` (float) the threshold on fitting R-squared to accept the result.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢PiPulseFreq

> Base *class*: `BaseAuto`

```python
a = quick.auto.PiPulseFreq(**kwargs)
```

Determine the qubit pulse frequency from Rabi, sweeping pi pulse frequency and find symmetry center.

### - PiPulseFreq.calibrate

```python
var, fig = a.calibrate(cycles=[], r=10, **kwargs)
```

Run the calibration.

**Parameters**:

- `cycles=[]` (list) extra cycles in Rabi. Cycle 0 (one pi pulse) will always be included.
- `r=10` (float) [MHz] single-side frequency range, centered by current `q_freq`.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢ReadoutFreq

> Base *class*: `BaseAuto`

```python
a = quick.auto.ReadoutFreq(**kwargs)
```

Determine the readout frequency from DispersiveSpectroscopy, maximizing the S21 separation.

### - ReadoutFreq.calibrate

```python
var, fig = a.calibrate(r=1, **kwargs)
```

Run the calibration.

**Parameters**:

- `r=1` (float) [MHz] single-side frequency range, centered by current `r_freq`.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢ReadoutState

> Base *class*: `BaseAuto`

```python
a = quick.auto.ReadoutState(**kwargs)
```

Calibrate `r_phase` and `r_threshold` for state distinguish.

### - ReadoutState.calibrate

```python
var, fig = a.calibrate(tol=0.1, **kwargs)
```

Run the calibration.

**Parameters**:

- `tol=0.1` (float) tolerance of visibility, below which the result will be rejected.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢Ramsey

> Base *class*: `BaseAuto`

```python
a = quick.auto.Ramsey(**kwargs)
```

Fine tune the qubit frequency by T2Ramsey.

### - Ramsey.calibrate

```python
var, fig = a.calibrate(fringe_freq=10, max_time=1, **kwargs)
```

Run the calibration.

**Parameters**:

- `fringe_freq=10` (float) [MHz] fringe frequency used in T2Ramsey
- `max_time=1` (float) [us] T2Ramsey maximum delay time.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢Readout

> Base *class*: `BaseAuto`

```python
a = quick.auto.Readout(**kwargs)
```

Use Nelder-Mead to optimize `r_power` and `r_length`

### - Readout.calibrate

```python
var, fig = a.calibrate()
```

Run the calibration.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢Relax

> Base *class*: `BaseAuto`

```python
a = quick.auto.Relax(**kwargs)
```

Estimate qubit relax time by T1. The scan will go to `0.8 * r_relax`. `r_relax` will be set as 5 times of measured T1.

### - Relax.calibrate

```python
var, fig = a.calibrate(**kwargs)
```

Run the calibration.

**Return**:

- `var` (dict|bool) `self.var` if succeeded. `False` if failed.
- `fig` generated plot

## 🟢run

```python
res = quick.auto.run(path, soccfg=None, soc=None, data_path=None)
```

Run an auto task for one step. To run the auto task till end, put this in a while loop.

**Parameters**:

- `path` (string) path to the auto task config file. The fill WILL be modified. See tutorial for details.
- `soccfg=None` QICK board connection config object. If None, the last connection will be used.
- `soc=None` QICK board connection socket object. If None, the last connection will be used.
- `data_path=None` (string) the data directory path. If None, data will not be saved.

**Returns**:

- `res` (bool) False for task completion, True for task incompletion.

**Examples**:

To run the auto task till end:

```python
while quick.auto.run("task.yml"):
    pass
    # clear_output() # clear output in Jupyter to avoid super long output
    # plt.show() # plot the figure from the auto steps.
```
