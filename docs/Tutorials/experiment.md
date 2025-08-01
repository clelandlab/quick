# Experiment

This is a tutorial of `quick.experiment`. For detailed parameters information, see API References.

`quick.experiment` use a variable dictionary to specify useful parameters for readout/qubit. The list of variables and there default values is [here](../API References/experiment/#var). For customization, see the last section of this document.

## Wiring and Pulse

`quick.experiment` uses the following channel convention by default. You can change them by changing variables.

|#|Channel|Purpose|
|---|---|---|
|`r0`|ADC 0|Readout acquisition|
|`g0`|DAC 0|Readout pulse|
|`g2`|DAC 2|Qubit pulse|

`quick.experiment` reserves the pulse index in range `0-5`. It uses the following pulse index by default. You can customize the pulse by passing in key-word arguments into experiment constructors.

|#|Purpose|
|---|---|
|`p0`|Readout pulse|
|`p1`|Qubit pi pulse|
|`p2`|Qubit half pi pulse|
|`p3`|Qubit half pi pulse with phase shift|

## Preparation

Connect to the QICK board, prepare the `data_path` and experiment variable

```python
import quick, yaml
import numpy as np
DATA_PATH = "path/to/data/directory/"
QICK_IP = "QICK board IP address"
soccfg, soc = quick.connect(QICK_IP)
v = dict(quick.experiment.var) # make a copy of variables
```

## LoopBack Test

Find the signal traveling time and update `v["r_offset"]`

```python
v["r_offset"] = 0
quick.experiment.LoopBack(var=v, data_path=DATA_PATH).run()
```

## Resonator Spectroscopy

Find the readout resonator. Do a broad scan first.

```python
quick.experiment.ResonatorSpectroscopy(
	var=v, data_path=DATA_PATH,
	title="broad scan",
	r_freq=np.arange(4000, 7000, 1)
).run()
```

> For most `quick.experiment`, you can pass in arbitrary sweep for variables. The sweeps will be in the order of the arguments.

Update `v["r_freq"]` and do a power spectroscopy.

```python
v["r_freq"] = 6000
quick.experiment.ResonatorSpectroscopy(
	var=v, data_path=DATA_PATH,
	title=f"PowerSpectroscopy {int(v['r_freq'])}",
	r_power=np.arange(-50, 0, 1), # first sweeping variable
	r_freq=np.arange(v["r_freq"] - 2, v["r_freq"] + 2, 0.05) # second sweeping variable
).run()
```

Find the sweet spot before the non-linear region and update `v["r_freq"]` and `v["r_power"]`

**Advanced**: measure spectrum using continuous pulse and long readout window like VNA:

```python
v["r_relax"] = 0.5
quick.experiment.ResonatorSpectroscopy(
  var=v, data_path=DATA_PATH,
  title=f"VNA-like {int(v['r_freq'])}",
  r_freq=np.arange(v["r_freq"] - 2, v["r_freq"] + 2, 0.05),
  # use continuous pulse and long readout window (213.33 us is the longest readout)
  p0_mode="periodic", r0_length=213, hard_avg=10
).run()
quick.experiment.LoopBack(var=v).run(silent=True) # stop the continuous pulse
```

## Qubit Spectroscopy

Play with `v["q_gain"]` and update `v["q_freq"]`.

```python
v["q_gain"] = 0.5
quick.experiment.QubitSpectroscopy(
	var=v, data_path=DATA_PATH,
	title=f"{int(v['r_freq'])}",
	q_freq=np.arange(3000, 4000, 1)
).run()
```

## Rabi Oscillation

Here is a length Rabi. Find the pi pulse and update `v["q_length"]`.

```python
v["q_gain"] = 1
v["r_relax"] = 500  # long relax time for qubit relax
quick.experiment.Rabi(
	var=v, data_path=DATA_PATH,
	title=f"length {int(v['r_freq'])}",
	q_length=np.arange(0.01, 0.4, 0.01)
).run()
```

You can also sweep `q_gain`(amplitude Rabi), `q_freq`(frequency Rabi), and even `cycle`(adding extra pi pulse cycles). See API References for details.

## IQ Scatter

Use your pi pulse to plot the IQ scatter to see the readout visibility and fidelity. Update `v["r_phase"]` and `v["r_threshold"]` for qubit state.

```python
for _ in range(2): # iterate
    data = quick.experiment.IQScatter(var=v).run().data.T
    phase, v["r_threshold"], visibility, Fg, Fe, c0, c1, fig = quick.iq_scatter(data[0] + 1j * data[1], data[2] + 1j * data[3])
    v["r_phase"] = (v["r_phase"] + phase) % 360
    plt.show()
```

You can also use the `DispersiveSpectroscopy` to find the best value for `v["r_freq"]`. See API References for details.

## T1

Measure, fit and plot T1 with your pi pulse and IQ center.

```python
data = quick.experiment.T1(
	var=v, data_path=DATA_PATH,
	title=f"{int(v['r_freq'])}",
	time=np.arange(0, 200, 5)
).run().data.T
popt, _, _, fig = quick.fitT1(data[0], data[1])
fig.show()
```

## T2

Measure, fit and plot T2 with your pi pulse and IQ center.

You can use either `T2Ramsey` or `T2Echo`. See API References for details.

```python
fringe_freq = 0.1
data = quick.experiment.T2Echo(
	var=v, data_path=DATA_PATH,
	title=f"{int(v['r_freq'])}",
	time=np.arange(0, 200, 1),
	cycle=3,
  fringe_freq=fringe_freq
).run().data.T
popt, _, _, fig = quick.fitT2(data[0], data[1], omega=2*np.pi*fringe_freq)
fig.show()
```

## Customization

To customize an experiment, there are several layers you can play with.

- Any variable can be sweeped by simply passing it as keyword argument into the experiment constructor.

- If you just want to slightly modify an existing experiment, use keyword argument to overwrite the [default programs](https://github.com/clelandlab/quick/blob/main/quick/constants/experiment.yml). For example, if you want a DRAG pi pulse instead of a gaussian pi pulse (`p1`) in Rabi and change the total repetition times to 3000, you can do:

```python
quick.experiment.Rabi(var=v, data_path=DATA_PATH, p1_style="DRAG", p1_delta=-180, rep=3000).run()
```

- If you want to write your own pulse sequence or scan function, you can inherit or use the `quick.BaseExperiment`. You can put a Mercator protocol into `quick.experiment.configs`. This requires some advanced understanding of Mercator and Experiment layers. See API References for details.

- If you don't want to inspect how `quick.experiment` works, it might also be applicable to go one layer down. You can directly use Mercator *class*. With the help of `quick.Sweep` and `quick.Saver` in the Helpers, you can basically do everything you want.

- If you want to do some very fancy QICK board manipulation, you can always `import qick` and use the original qick firmware. `quick.Sweep` and `quick.Saver` will also be helpful.
