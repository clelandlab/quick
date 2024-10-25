# Mercator

A Python interface to run Mercator protocol. See Tutorials for details of Mercator protocol.

> Only Mercator *(class)* is exposed as `quick.Mercator`.

## 🔵Mercator

> Base *class*: `qick.asm_v2。AveragerProgramV2`

```python
m = quick.Mercator(soccfg, cfg)
```

The Mercator class to run Mercator protocol.

**Parameters**:

- `soccfg` QICK board socket config object.
- `cfg` (dict) Mercator protocol in a Python dictionary. **Mercator will modify this dictionary, including flattening steps and evaluating gains.**

> Methods including `Mercator.initialize` and `Mercator.body` should never be used manually. Thus, they are not documented here.

### 🟡Mercator.c

```python
m.c
```

Internal control object generated from the Mercator protocol. Mostly for internal or debug use. It should never be changed manually.

### 🔵Mercator.acquire

```python
I, Q = m.acquire(soc, progress=False)
```

Acquire data. Data in each acquisition window are averaged into one data point.

**Parameters**:

- `soc` QICK board socket object.
- `progress=False` whether to show progress bar.

**Return**:

- `I` (np.ndarray) I data, shape depends on `rep` and triggering.
- `Q` (np.ndarray) Q data, same shape as `I`.

### 🔵Mercator.acquire_decimated

```python
I, Q = m.acquire_decimated(soc, progress=False)
```

Acquire time-series data.

**Parameters**:

- `soc` QICK board socket object.
- `progress=False` whether to show progress bar.

**Return**:

- `I` (np.ndarray) I data, including time-series dimension.
- `Q` (np.ndarray) Q data, same shape as `I`.

### 🔵Mercator.light

```python
m.light()
```

Plot the pulse sequence. Only `idata` will be plotted, and frequency and phase information are not shown on the plot. See [Mercator Protocol Tutorial](../../Tutorials/mercator/) for an example.
