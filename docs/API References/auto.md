# Auto

Automatic scripts for qubit measurements. **Under active development.**

> Everything exposed to `quick.auto`.

## 🔵BaseAuto

```python
a = quick.auto.BaseAuto(var)
```

General base *class* for other auto scripts.

**Parameters**:

- `var` experimental variables. It will NOT be modified.

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

## 🟢Resonator

> Base *class*: `BaseAuto`

Determine the readout power and readout frequency from PowerSpectroscopy.

### - Resonator.measure

```python
a.measure(silent=False, data_path=None, soccfg=None, soc=None)
```

Run experiment to measure data.

**Parameters**:

- `silent=False` (bool) whether to avoid any printing.
- `data_path=None` (str) directory to save data. If not provided, data will not be saved to file.
- `soccfg=None` QICK board socket config object. If not provided, the last connected one will be used by calling `quick.getSoc()`.
- `soc=None` QICK board socket object.

### - Resonator.calibrate

```python
a.calibrate(silent=False):
```

Calibrate the experiment variables from data.

**Parameters**:

- `silent=False` (bool) whether to avoid any printing.

### - Resonator.update

```python
a.update(v):
```

Update relavent variables (`r_freq`, `r_power`)

**Parameters**:

- `v` (dict) experiment variable to be updated. **Will be modified!**

