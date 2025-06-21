# Mercator

A Python interface to run Mercator protocol. See Tutorials for details of Mercator protocol.

> Only Mercator *(class)* is exported as `quick.Mercator`.

## 🔵Mercator

> Base *class*: `qick.asm_v2.AveragerProgramV2`

```python
m = quick.Mercator(soccfg, cfg)
```

The Mercator class to run Mercator protocol.

**Parameters**:

- `soccfg` QICK board socket config object.
- `cfg` (dict) Mercator protocol in a Python dictionary. **Mercator will modify this dictionary, including flattening steps and evaluating gains.**

### 🔵Mercator.acquire

```python
I, Q = m.acquire(soc, progress=False)
```

Acquire data. Data in each readout acquisition window are averaged into one data point.

**Parameters**:

- `soc` QICK board socket object.
- `progress=False` whether to show progress bar.

**Return**:

- `I` (np.ndarray) I data, at least has 2 dimensions. Axis 0 matches the index (**NOT channel number**) of defined readout channels. Axis 1 matches the index of triggering. If `rep` is not 0, axis 2 will be the repetition index.
- `Q` (np.ndarray) Q data, same shape as `I`.

### 🔵Mercator.acquire_decimated

```python
I, Q = m.acquire_decimated(soc, progress=False)
```

Acquire time-series data. Use `m.get_time_axis(ro_index)` to get the time data in us.

**Parameters**:

- `soc` QICK board socket object.
- `progress=False` whether to show progress bar.

**Return**:

- `I` (np.ndarray) I data, same shape as returned by `acquire`, plus an extra time-series dimension.
- `Q` (np.ndarray) Q data, same shape as `I`.

### 🔵Mercator.light

```python
m.light()
```

Plot the pulse sequence. Only `idata` will be plotted, and frequency and phase information are not shown on the plot. See [Mercator Protocol Tutorial](../../Tutorials/mercator/) for an example.
