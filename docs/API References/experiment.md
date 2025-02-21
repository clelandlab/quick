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
r_power: -30      # [dB] readout pulse power
r_length: 2       # [us] readout pulse length
r_phase: 0        # [deg] readout pulse phase
r_offset: 0       # [us] readout window offset
r_threshold: 0    # threshold, above which is 1-state
r_reset: 0        # [us] wait time for qubit reset (active reset).
r_relax: 1        # [us] readout relax time
q_freq: 5000      # [MHz] qubit pulse frequency
q_length: 2       # [us] qubit pulse length
q_delta: -180     # [MHz] qubit anharmonicity
q_gain: 1         # [-1, 1] qubit pulse (pi pulse) gain
q_gain_2: 0.5     # [-1, 1] half pi pulse gain
```

## 🟡configs

```python
quick.experiment.configs
```

Global config templates used by `BaseExperiment`. These are default Mercator protocols for the experiments. It is loaded from [here](https://github.com/clelandlab/quick/blob/main/quick/experiment.yml). Mostly internal use.

All of the programs are saved as strings for variable insersion. The experiment variables will be inserted into the Mercator protocol templates with `quick.evalStr`.

## 🔵BaseExperiment

```python
e = quick.experiment.BaseExperiment(data_path=None, title="", soccfg=None, soc=None, var=None, **kwargs)
```

General base *class* for other experiments using Mercator protocol. If not overwriten, all the other experiment *class* have the properties and methods described here. If not specified, all `**kwargs` in the other experiment *class* are passed into this constructor.

**Parameters**:

- `data_path=None` (str) directory to save data. If not provided, data will not be saved to file.
- `title=""` (str) filename of data. A prefix of experiment name will be added to it, eg. `(BaseExperiment)your title`.
- `soccfg=None` QICK board socket config object. If not provided, the last connected one will be used by calling `quick.getSoc()`.
- `soc=None` QICK board socket object.
- `var=None` experimental variables to be inserted into the corresponding Mercator protocol template of the experiment (in `quick.experiment.configs`) and pre-defined experiment-specific variables. It will NOT be modified. The default value in `quick.experiment.var` will be used if any keys are missing.
- `**kwargs` other keyword arguments. Any keys in `self.var` can be used as variable overwriting(by a value) or sweeping(by a list). The sweeping order is determined by the order of keyword arguments. Keys not in `self.var` are used to overwrite Mercator protocol and therefore overwrite the pulse sequence. The overwriting happens after variable insertions.

**Details**: (Advanced: How `BaseExperment` works?)

It will gather some necessary information first:

- a. the template program (`quick.experiment.configs`)
- b. the template variable (`quick.experiment.var`)
- c. new variable defined before calling `__init__`, like `time` in `T1`
- d. input variable (`var` argument)
- e. other keyword arguments that has the same key as in (b), (c), (d) combined.
- f. other keyword arguments that does not belong to (e), and not predefined (predefined: `data_path` etc).
- g. internal variable (`self._var`)

The logic of execution is:

1. In `__init__`, iterables in (e) will be stored as `self.sweep`, initial values and non-iterables in (e) will overwrite (d). Then (d) will overwrite (c) and then overwrite (b), forming `self.var`.
2. `self.eval_config` will evaluate the Mercator protocol by inserting some variables (eg. `self.var`) with (g) into (a), forming `self.config`. Then (f) will overwrite `self.config`.
3. During most runs, `self.sweep` will be performed by `quick.Sweep`. In each loop, a set of variable will be generated from `self.var`, and `self.eval_config` is called to generate the Mercator protocol config for that specific loop run.

### - BaseExperiment.key

```python
e.key
```

Same as experiment class name. Used to obtain config template by `quick.experiment.configs[self.key]`. Will be saved in data metainformation as `params["quick.experiment"]`.

### - BaseExperiment.var

```python
e.var
```

The experiment variable dictionary used in the experiment. Variables are inserted into the Mercator protocol of the experiment by `quick.evalStr` in each sweep/run.

### - BaseExperiment._var

```python
e._var
```

The internal variable dictionary used in the experiment. This dictionary will be used as global variables in `self.eval_config`, but it will not be saved or modified by user input. Thus, you can use it as internal variables of the experiment for further flexibility.

### - BaseExperiment.var_label

```python
e.var_label
```

The experiment variable label dictionary used in the experiment. This determines the data label and unit.

### - BaseExperiment.data

```python
e.data
```

Data acquired by the experiment. Same structure as saved by the data saver `quick.Saver`. It is just a copy of data saved in the memory.

### - BaseExperiment.config

```python
e.config
```

Program config (pulse sequence) dictionary in Mercator protocol.

### - BaseExperiment.config_update

```python
e.config_update
```

Config update dictionary. Used to overwrite `self.config` after variable insertion in each sweep/run.

### - BaseExperiment.sweep

```python
e.sweep
```

Sweep dictionary. Store sweeping variables.

### - BaseExperiment.m

```python
e.m
```

The Mercator instance created by the experiment.

### - BaseExperiment.eval_config

```python
e.eval_config(v)
```

Perform variable insertion and then config overwriting, generating `self.config` from the template `quick.experiment.configs[self.key]`. This function use `self._var` as global variables when evaluating strings.

**Parameters**:

- `v` (dict) variable dictionary, used in variable insertion.

### - BaseExperiment.prepare

```python
e.prepare(indep_params=[], dep_params=[], dB=False, population=False)
```

prepare the standard S21 measurements (amplitude, phase, I, Q), create data saver. Mostly for internal use.

**Parameters**:

- `indep_params=[]` (list) a list of 2-tuples, specifying meta information for independent variables, in the format of `("Name", "Unit")`. Note that variables in `self.sweep` are automatically added without passing in.
- `dep_params=[]` (list) a list of 2-tuples, specifying meta information for dependent variables, in the format of `("Name", "Unit")`. If empty, default values will be generated according to the following arguments.
- `dB=False` (bool) whether to measure amplitude in log scale (dB with normalization).
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
e.acquire_S21(indep_list, dB=False, decimated=False, population=False)
```

acquire data for standard S21 measurement. Run the program specified by `self.config` in Mercator protocol. Mostly for internal use.

**Parameters**:

- `indep_list` (list) a list of values for independent variables, in the exact order as specified by `prepare`.
- `dB=False` (bool) whether to measure amplitude in log scale (dB with normalization).
- `decimated=False` (bool) whether to acquire for time-series output.
- `population=False` (bool) whether to measure the qubit population.

### = BaseExperiment.run

```python
e = e.run(silent=False, dB=False, population=False)
```

Run the experiment. See details below.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
- `dB=False` (bool) whether to measure amplitude in log scale (dB with normalization).
- `population=False` (bool) whether to measure the qubit population.

**Return**:

- `e` the experiment object itself.

**Detail**:

This method performs the following steps:

1. Call `self.prepare` to prepare the data saver.
2. Use `quick.Sweep` to sweep the variables in `self.sweep`. In each sweep:
    1. Call `self.eval_config` to generate the corresponding `self.config`
    2. Call `self.acquire_S21` to execute and save data with `quick.Mercator`
3. Call and return `self.conclude`

### - BaseExperiment.conclude

```python
e = e.conclude(silent=False)
```

finalize the experiment, print message for completion. Mostly for internal use.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

**Return**:

- `e` the experiment object itself.

### - BaseExperiment.light

```python
e = e.light()
```

Light the internal Mercator object.

**Return**:

- `e` the experiment object itself.

## 🟢LoopBack

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.LoopBack(**kwargs)
```

Measure the loop-back signal. No variable sweeping.

- `indep_params = [("Time", "us")]`
- `dep_params = [("Amplitude", ""), ("Phase", "rad"), ("I", ""), ("Q", "")]`

### - LoopBack.run

```python
e = e.run(silent=False, dB=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
- `dB=False` (bool) whether to measure amplitude in log scale (dB with normalization).

**Return**:

- `e` the experiment object itself.

## 🟢ResonatorSpectroscopy

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.ResonatorSpectroscopy(**kwargs)
```

Measure the resonator spectroscopy, including the power spectroscopy.

- Arbitrary variable sweeping
- `dep_params = [("Amplitude", "dB"), ("Phase", "rad"), ("I", ""), ("Q", "")]`

### - ResonatorSpectroscopy.run

```python
e = e.run(silent=False, dB=True)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
- `dB=True` (bool) whether to measure amplitude in log scale (dB with normalization).

**Return**:

- `e` the experiment object itself.

## 🟢QubitSpectroscopy

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.QubitSpectroscopy(**kwargs)
```

Measure the qubit spectroscopy, or two-tone spectroscopy.

- Arbitrary variable sweeping
- `dep_params = [("Amplitude", ""), ("Phase", "rad"), ("I", ""), ("Q", "")]`
    - include `(Population, "")` by `e.run(population=True)`
    - use log magnitude by `e.run(dB=True)`

## 🟢Rabi

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.Rabi(**kwargs)
```

Measure the Rabi oscillation.

- Arbitrary variable sweeping, plus:
    - `cycle=0` (int) extra pi pulse cycle. Every cycle gives two extra pi pulse. `cycle=0` gives 1 pi pulse.
- `dep_params = [("Population", ""), ("Amplitude", ""), ("Phase", "rad"), ("I", ""), ("Q", "")]`
    - remove `(Population, "")` by `e.run(population=False)`

### - Rabi.run

```python
e.run(silent=False, population=True)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
- `population=True` (bool) whether to measure the qubit population.

**Return**:

- `e` the experiment object itself.

## 🟢IQScatter

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.IQScatter(**kwargs)
```

Measure the IQ scatter data. No variable sweeping.

- `dep_params = [("I 0", ""), ("Q 0", ""), ("I 1", ""), ("Q 1", "")]`

### - IQScatter.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

**Return**:

- `e` the experiment object itself.

## 🟢IQTrace

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.IQTrace(**kwargs)
```

Measure the IQ scatter data with hard average.

- Arbitrary variable sweeping, plus:
    - `rr_length=0.1` (us) readout window length (independent from `r_length` here).
- `dep_params = [("I 0", ""), ("Q 0", ""), ("I 1", ""), ("Q 1", "")]`

### - IQTrace.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

**Return**:

- `e` the experiment object itself.

## 🟢DispersiveSpectroscopy

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.DispersiveSpectroscopy(r_freq=[], **kwargs)
```

Measure the dispersive spectroscopy.

- Arbitrary variable sweeping
- `dep_params = [("Amplitude 0", "dB"), ("Phase 0", "rad"), ("I 0", ""), ("Q 0", ""), ("Amplitude 1", "dB"), ("Phase 1", "rad"), ("I 1", ""), ("Q 1", "")]`

**Parameters**:

- `r_freq=[]` (1D ArrayLike) resonator frequency list `("Frequency", "MHz")`.

### - DispersiveSpectroscopy.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

**Return**:

- `e` the experiment object itself.

## 🟢T1

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.T1(**kwargs)
```

Measure the T1 decay.

- Arbitrary variable sweeping, plus:
    - `time=0` (us) readout delay time.
- `dep_params = [("Population", ""), ("Amplitude", ""), ("Phase", "rad"), ("I", ""), ("Q", "")]`
    - remove `(Population, "")` by `e.run(population=False)`

### - T1.run

```python
e.run(silent=False, population=True)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
- `population=True` (bool) whether to measure the qubit population.

**Return**:

- `e` the experiment object itself.

## 🟢T2Ramsey

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.T2Ramsey(**kwargs)
```

Measure the T2 decay with fringe by Ramsey oscillation.

- Arbitrary variable sweeping, plus:
    - `time=0` (us) readout delay time.
    - `fringe_freq=0` (MHz) fringe frequency.
- `dep_params = [("Population", ""), ("Amplitude", ""), ("Phase", "rad"), ("I", ""), ("Q", "")]`
    - remove `(Population, "")` by `e.run(population=False)`

### - T2Ramsey.run

```python
e.run(silent=False, population=True)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
- `population=True` (bool) whether to measure the qubit population.

**Return**:

- `e` the experiment object itself.

## 🟢T2Echo

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.T2Echo(**kwargs)
```

Measure the T2 decay with fringe by Hahn echo or CPMG method. Pi pulses for echo will always be on +y axis (90 degrees phase).

- Arbitrary variable sweeping, plus:
    - `time=0` (us) readout delay time.
    - `cycle=0` (int) extra cycle in the CPMG method. Each extra cycle introduces 1 extra pi pulse, implementing the CPMG pulse sequence. `cycle=0` gives 1 pi pulse.
    - `fringe_freq=0` (MHz) fringe frequency.
- `dep_params = [("Population", ""), ("Amplitude", ""), ("Phase", "rad"), ("I", ""), ("Q", "")]`
    - remove `(Population, "")` by `e.run(population=False)`

### - T2Echo.run

```python
e.run(silent=False, population=True)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.
- `population=True` (bool) whether to measure the qubit population.

**Return**:

- `e` the experiment object itself.

## 🔵Random

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.Random(**kwargs)
```

Generate a sequence of random bit by a qubit! No variable sweeping.

### - Random.run

```python
e.run(silent=False)
```

run the experiment. `e.data` will be a list of random bits. It's guaranteed to be balanced, and the bit number is roughly 1/4 of the `rep`.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

**Return**:

- `e` the experiment object itself.

### - Random.random

```python
a = e.random(silent=True)
```

generate a float number within range (0, 1) uniformly.

**Parameters**:

- `silent=True` (bool) Whether to avoid any printing.

**Return**:

- `a` (float) a random number between 0 and 1.

## 🟢QND

> Base *class*: `BaseExperiment`

```python
e = quick.experiment.QND(**kwargs)
```

QNDness measurement by randomized Pi pulse and repeated readout.

- Arbitrary variable sweeping, plus:
    - `cycle=10` (int) cycles of randomized Pi pulse and readout.
    - `random=100` (int) number of randomization.
- `dep_params = [("Correlation", "")]`

### - QND.run

```python
e.run(silent=False)
```

run the experiment.

**Parameters**:

- `silent=False` (bool) Whether to avoid any printing.

**Return**:

- `e` the experiment object itself.


