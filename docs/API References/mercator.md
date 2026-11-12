# Mercator

A Python interface to run Mercator protocol. See Tutorials for details of Mercator protocol.

> Only Mercator *(class)* is exported as `quick.Mercator`.

## 🔵Mercator

> This checkout uses a local scalar tProc v2 compiler and acquisition backend. It does not inherit from or import QICK's `AveragerProgramV2`. Mercator dictionaries retain their existing protocol syntax; arbitrary inherited QICK compiler APIs and sweep objects are not supported.

```python
m = quick.Mercator(soccfg, cfg)
```

The Mercator class to run Mercator protocol.

**Parameters**:

- `soccfg` Local board configuration object returned by `quick.connect()`.
- `cfg` (dict) Mercator protocol in a Python dictionary. **Mercator will modify this dictionary, including flattening steps and evaluating gains.**

### 🔵Mercator.acquire

```python
I, Q = m.acquire(soc, progress=False)
```

Acquire data. Data in each readout acquisition window are averaged into one data point.

**Parameters**:

- `soc` QuICK TCP client.
- `progress=False` whether to show progress bar.
- `timeout=None` Optional deadline in seconds for each software round; independent of the connection's network timeout.
- `remove_offset=True`, `threshold=None`, `angle=None` preserve the acquisition normalization/population-classification options. Threshold equality maps to zero after software rotation/normalization; FPGA conditional pulses use their separate raw-I comparison.

**Return**:

- `I` (np.ndarray) I data, with at least 2 dimensions. Axis 0 matches the index (**NOT channel number**) of defined readout channels. Axis 1 matches the trigger index. If `rep` is positive, axis 2 is the retained repetition index.
- `Q` (np.ndarray) Q data, same shape as `I`.

### 🔵Mercator.acquire_decimated

```python
I, Q = m.acquire_decimated(soc, progress=False)
```

Acquire finite time-series data. Use `m.get_time_axis(ro_index)` for sample times in µs. Each window must be positive and smaller than its buffer; total samples across hardware shots/triggers must fit the buffer. Multiple smaller windows may collectively fill it exactly.

**Parameters**:

- `soc` QuICK TCP client.
- `progress=False` whether to show progress bar.
- `timeout=None` Optional per-software-round deadline; `remove_offset=True` controls sample-offset subtraction.

**Return**:

- `I` (np.ndarray) Axis 0 follows declared readouts; the last axis is sample time. Hardware loops are flattened into a shot count `hard_avg * max(rep, 1)`. With multiple shots and multiple triggers, intervening axes are shot then trigger. If only one of those counts exceeds one, there is one intervening axis; when both are one, neither appears. Software rounds are averaged. Channels must have compatible output shapes, as before.
- `Q` (np.ndarray) Q data, same shape as `I`.

### 🔵Mercator.light

```python
m.light()
```

Plot the pulse sequence. Only `idata` will be plotted, and frequency and phase information are not shown on the plot. See [Mercator Protocol Tutorial](../Tutorials/mercator.md) for an example.
