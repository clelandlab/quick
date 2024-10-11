# Mercator

A Python interface to run Mercator protocol. See Tutorials for details of Mercator protocol.

> Only Mercator *(class)* is exposed as `quick.Mercator`.

## 🔵Mercator

> Base *class*: `qick.NDAveragerProgram`

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
m.acquire(soc)
```

Acquire averaged data. Same as `qick.NDAveragerProgram.acquire`.

**Parameters**:

- `soc` QICK board socket object.

### 🔵Mercator.acquire_decimated

```python
m.acquire_decimated(soc, progress=False)
```

Acquire time-series data. Similar to `qick.NDAveragerProgram.acquire_decimated`.

**Parameters**:

- `soc` QICK board socket object.
- `progress=False` whether to show progress bar.

### 🔵Mercator.light

```python
m.light()
```

Plot the pulse sequence. Only `idata` will be plotted, and frequency and phase information are not shown on the plot.
