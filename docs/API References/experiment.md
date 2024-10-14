# Experiment

Useful experiments for qubit measurements.

> Everything exposed to `quick.experiment`.

## 🟢var

```python
quick.experiment.var
```

Global experiment variable. Serve as a template to generate variable dictionary.

**Example**:

```python
v = dict(quick.experiment.var) # a copy of experiment variables.
```

**Details**:

```yaml
rr: 0             # readout channel
r: 0              # generator channel for readout pulse
q: 2              # generator channel for qubit pulse
r_freq: 5000      # [MHz] readout pulse frequency
r_power: -30      # [dBm] readout pulse power
r_length: 2       # [us] readout pulse length
r_phase: 0        # [deg] readout pulse phase
r_offset: 0       # [us] readout window offset
r_balun: 3        # readout pulse balun
r_threshold: 0    # threshold, above which is 1-state
r_relax: 1        # [us] readout relax time
q_freq: 5000      # [MHz] qubit pulse frequency
q_length: 2       # [us] qubit pulse length
q_delta: -180     # [MHz] qubit anharmonicity
q_gain: 30000     # [0-32766] qubit pulse (pi pulse) gain
q_gain_2: 15000   # [0-32766] half pi pulse gain
q_T1: 80          # [us] qubit T1
fringe_freq: 1    # [MHz] fringe frequency in T2Ramsey and T2Echo
active_reset: 0   # [us] active reset wait time. 0 for disable.
```

## 🟡configs

```python
quick.experiment.configs
```

Global config templates used by `BaseExperiment`. These are default Mercator protocols for the experiments. It is loaded from [here](https://github.com/clelandlab/quick/blob/main/quick/constants/experiment.yml). Mostly internal use.

All of the programs are saved as strings for variable insersion. The experiment variables will be inserted into the Mercator protocol templates with `quick.evalStr`.

## 🟡BaseExperiment

```python
e = quick.experiment.BaseExperiment(data_path=None, title="", soccfg=None, soc=None, var=None, **kwargs)
```

General base *class* for other experiments using Mercator protocol. Mostly for internal use.

**Parameters**:

- `data_path=None` (str) directory to save data. If not provided, data will not be saved to file.
- `title=""` (str) filename of data. A prefix of experiment name will be added to it, eg. `(BaseExperiment)your title`.
- `soccfg=None` QICK board socket config object. If not provided, the last connected one will be used by calling `quick.getSoc()`.
- `soc=None` QICK board socket object.
- `var=None` experimental variables to be inserted into the corresponding Mercator protocol template of the experiment (in `quick.experiment.configs`). It will NOT be modified. The default value in `quick.experiment.var` will be used if any keys are missing.
- `**kwargs` other keyword arguments. Can be used to overwrite Mercator protocol and therefore overwrite the pulse sequence. The overwriting happens after variable insertions.

### - BaseExperiment.key

```python
e.key
```

Same as experiment class name. Will be saved as `params["quick"]`.

### - BaseExperiment.var

```python
e.var
```

The experiment variable dictionary used by the experiment.  Will be saved as `params["var"]`.

### - BaseExperiment.data

```python
e.data
```

Data acquired by the experiment. Same structure as saved by the data saver `quick.Saver`. It is just a copy of data saved in the memory.

### - BaseExperiment.config

```python
e.config
```

Program config template in Mercator protocol.

### - BaseExperiment.m

```python
e.m
```

The Mercator instance created by the experiment.

### - BaseExperiment.prepare

```python
e.prepare(indep_params=[], log_mag=False, population=False)
```

prepare the standard S21 measurements (amplitude, phase, I, Q), create data saver. Mostly for internal use.

**Parameters**:

- `indep_params=[]` (list) a list of 2-tuples, specifying meta information for independent variables, in the format of `("Name", "Unit")`
- `log_mag=False` (bool) whether to measure amplitude in log scale (with normalization).
- `population=False` (bool) whether to measure the qubit population.

### - BaseExperiment.add_data

```python
e.add_data(data)
```

add and save data. Mostly for internal use.

**Parameters**:

- `data` (2D ArrayLike) a list of data rows. Each row should be a list of numerical data, in the exact order defined in `indep_params` and `dep_params`.

### - BaseExperiment.acquire_S21

```python
e.acquire_S21(cfg, indep_list, log_mag=False, decimated=False, population=False)
```

acquire data for standard S21 measurement. Mostly for internal use.

**Parameters**:

- `cfg` (dict) a program in Mercator protocol.
- `indep_list` (list) a list of values for independent variables, in the exact order as specified by `prepare`.
- `log_mag=False` (bool) whether to measure amplitude in log scale (with normalization).
- `decimated=False` (bool) whether to acquire for time-series output.
- `population=False` (bool) whether to measure the qubit population.

### - BaseExperiment.conclude

```python
e.conclude(silent=False)
```

finalize the experiment, print message for completion. Mostly for internal use.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

### - BaseExperiment.light

```python
e.light()
```

Light the internal Mercator object.

## 🟢LoopBack

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.LoopBack(**kwargs)
```

Measure the loop-back signal.

- `indep_params = [("Time", "us")]`
- `dep_params = [("Amplitude", "", "lin mag"), ("Phase", "rad"), ("I", ""), ("Q", "")]`

### - LoopBack.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

## 🟢ResonatorSpectroscopy

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.ResonatorSpectroscopy(r_freqs=[], r_powers=None, **kwargs)
```

Measure the resonator spectroscopy, including the power spectroscopy.

- `dep_params = [("Amplitude", "dB", "log mag"), ("Phase", "rad"), ("I", ""), ("Q", "")]`
- Amplitude can be `lin mag` by setting `log_mag=False` in parameters of `e.run()`

**Parameters**:

- `r_freqs=[]` (1D ArrayLike) resonator frequency list `("Frequency", "MHz")`.
- `r_powers=None` (1D ArrayLike) optional resonator power list `("Power", "dBm")`.

### - ResonatorSpectroscopy.run

```python
e.run(silent=False, log_mag=True)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
- `log_mag=True` (bool) Whether to use log magnitude (dB).

## 🟢QubitSpectroscopy

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.QubitSpectroscopy(q_freqs=[], r_freqs=None, **kwargs)
```

Measure the qubit spectroscopy, or two-tone spectroscopy.

- `dep_params = [("Amplitude", "", "lin mag"), ("Phase", "rad"), ("I", ""), ("Q", "")]`

**Parameters**:

- `q_freqs=[]` (1D ArrayLike) qubit frequency list `("Qubit Frequency", "MHz")`.
- `r_freqs=None` (1D ArrayLike) optional resonator frequency list `("Readout Frequency", "MHz")`.

### - QubitSpectroscopy.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

## 🟢Rabi

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.Rabi(q_lengths=None, q_gains=None, q_freqs=None, cycles=None, **kwargs)
```

Measure the Rabi oscillation. Any combinations of the four independent variables can be sweeped.

- `dep_params = [("Population", ""), ("Amplitude", "", "lin mag"), ("Phase", "rad"), ("I", ""), ("Q", "")]`

**Parameters**:

- `q_lengths=None` (1D ArrayLike) optional qubit pulse length list `("Pulse Length", "us")`.
- `q_gains=None` (1D ArrayLike) optional qubit pulse gain list `("Pulse Gain", "a.u.")`.
- `q_freqs=None` (1D ArrayLike) optional qubit frequency list `("Qubit Frequency", "MHz")`.
- `cycles=None` (1D ArrayLike) optional extra cycle list `("Extra Cycles", "")`. Each extra cycle introduces 2 extra pi pulses. `cycle=0` gives one pi pulse.

### - Rabi.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

## 🟢IQScatter

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.IQScatter(**kwargs)
```

Measure the IQ scatter data.

- `dep_params = [("I 0", ""), ("Q 0", ""), ("I 1", ""), ("Q 1", "")]`

### - IQScatter.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

## 🔵ActiveReset

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.ActiveReset(**kwargs)
```

Test active reset. Performing an amplitude Rabi for 100 points.

- `dep_params = [("Population", "", "before reset"), ("Population", "", "after reset")]`

### - ActiveReset.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

## 🟢DispersiveSpectroscopy

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.DispersiveSpectroscopy(r_freqs=[], **kwargs)
```

Measure the dispersive spectroscopy.

- `dep_params = [("Amplitude 0", "dB", "log mag"), ("Phase 0", "rad"), ("I 0", ""), ("Q 0", ""), ("Amplitude 1", "dB", "log mag"), ("Phase 1", "rad"), ("I 1", ""), ("Q 1", "")]`

**Parameters**:

- `r_freqs=[]` (1D ArrayLike) resonator frequency list `("Frequency", "MHz")`.

### - DispersiveSpectroscopy.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

## 🟢T1

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.T1(times=[], **kwargs)
```

Measure the T1 decay.

- `dep_params = [("Population", ""), ("Amplitude", "", "lin mag"), ("Phase", "rad"), ("I", ""), ("Q", "")]`

**Parameters**:

- `times=[]` (1D ArrayLike) pulse delay time list `("Pulse Delay", "us")`.

### - T1.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

## 🟢T2Ramsey

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.T2Ramsey(times=[], fringe_freqs=None, **kwargs)
```

Measure the T2 decay with fringe by Ramsey oscillation.

- `dep_params = [("Population", ""), ("Amplitude", "", "lin mag"), ("Phase", "rad"), ("I", ""), ("Q", "")]`

**Parameters**:

- `times=[]` (1D ArrayLike) pulse delay time list `("Pulse Delay", "us")`.
- `fringe_freqs=None` (1D ArrayLike) fringe frequency list `("Fringe Frequency", "MHz")`.

### - T2Ramsey.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

## 🟢T2Echo

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.T2Echo(times=[], fringe_freqs=None, cycle=0, **kwargs)
```

Measure the T2 decay with fringe by Hahn echo or CPMG method. Pi pulses for echo will always be on +y axis (90 degrees phase).

- `dep_params = [("Population", ""), ("Amplitude", "", "lin mag"), ("Phase", "rad"), ("I", ""), ("Q", "")]`

**Parameters**:

- `times=[]` (1D ArrayLike) pulse delay time list `("Pulse Delay", "us")`.
- `fringe_freqs=None` (1D ArrayLike) fringe frequency list `("Fringe Frequency", "MHz")`.
- `cycle=0` (int) extra cycle in the CPMG method. Each extra cycle introduces 1 extra pi pulse, implementing the CPMG pulse sequence. `cycle=0` gives 1 pi pulse.

### - T2Echo.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
