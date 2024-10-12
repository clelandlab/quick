# Experiment

This is a tutorial of `quick.experiment`. For detailed parameters information, see API References.

> Note in the front: `quick.experiment` use a variable dictionary to specify useful parameters for readout/qubit. For customization, see the last section of this document.

## Wiring

`quick.experiment` follows a fixed convention in wire connection. You must connect the sample to specific qick board channels.

|#|Channel|Purpose|
|---|---|---|
|`r0`|ADC 0|Readout acquisition|
|`g0`|DAC 0|Readout pulse|
|`g2`|DAC 2|Qubit pulse|

## Preparation

Connect to the QICK board, prepare the `data_path` and experiment variable

```python
import quick, yaml
import numpy as np
DATA_PATH = "path/to/data/directory/"
soccfg, soc = quick.connect("QICK board IP address")
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
	r_freqs=np.arange(4000, 7000, 1)
).run()
```

Update `v["r_freq"]` and do a power spectroscopy.

```python
v["r_freq"] = 6000
quick.experiment.ResonatorSpectroscopy(
	var=v, data_path=DATA_PATH,
	title=f"PowerSpectroscopy {int(v['r_freq'])}",
	r_freqs=np.arange(v["r_freq"] - 2, v["r_freq"] + 2, 0.05),
	r_powers=np.arange(-65, -15, 1)
).run()
```

Find the sweet spot before the non-linear region and update `v["r_freq"]` and `v["r_power"]`

## Qubit Spectroscopy

Play with `v["q_gain"]` and update `v["q_freq"]`.

```python
v["q_gain"] = 10000
quick.experiment.QubitSpectroscopy(
	var=v, data_path=DATA_PATH,
	title=f"{int(v['r_freq'])}",
	q_freqs=np.arange(3000, 4000, 1)
).run()
```

## Rabi Oscillation

Here is a length Rabi. Find the pi pulse and update `v["q_length"]`.

```python
v["q_gain"] = 30000
v["r_relax"] = 500  # long relax time for qubit relax
quick.experiment.Rabi(
	var=v, data_path=DATA_PATH,
	title=f"length {int(v['r_freq'])}",
	q_lengths=np.arange(0.01, 0.4, 0.01)
).run()
```

You can also do amplitude rabi, frequency rabi, and even adding more pi pulse cycles. See API References for details.

## IQ Scatter

Use your pi pulse to plot the IQ scatter to see the readout visibility and fidelity. Update `v["r_phase"]` and `v["r_threshold"]` for qubit state.

```python
v["r_phase"] = 0
data = quick.experiment.IQScatter(var=v).run().data.T
c0, c1, visibility, Fg, Fe, fig = quick.iq_scatter(data[0] + 1j * data[1], data[2] + 1j * data[3])
v["r_phase"], v["r_threshold"] = quick.iq_rotation(c0, c1)
fig.show()
```

You can also use the `DispersiveSpectroscopy` to find the best value for `v["r_freq"]`. See API References for details.

## T1

Measure, fit and plot T1 with your pi pulse and IQ center, update `v["q_T1"]`

```python
data = quick.experiment.T1(
	var=v, data_path=DATA_PATH,
	title=f"{int(v['r_freq'])}",
	times=np.arange(0, 200, 5)
).run().data.T
popt, _, _, fig = quick.fitT1(data[0], data[1])
v["q_T1"] = float(popt[1])
fig.show()
```

## T2

Measure, fit and plot T2 with your pi pulse and IQ center. Set fringe frequency `v["fringe_freq"]` first.

You can use either `T2Ramsey` or `T2Echo`. See API References for details.

```python
v["fringe_freq"] = 0.1
data = quick.experiment.T2Echo(
	var=v, data_path=DATA_PATH,
	title=f"{int(v['r_freq'])}",
	times=np.arange(0, 200, 1),
	cycle=3
).run().data.T
popt, _, _, fig = quick.fitT2(data[0], data[1], omega=2*np.pi*v["fringe_freq"])
fig.show()
```

## Customization

To customize an experiment, there are several layers you can play with.

- If you just want to slightly modify an existing experiment, use keyword argument to overwrite the [default programs](https://github.com/clelandlab/quick/blob/main/quick/constants/experiment.yml). For example, if you want a DRAG pi pulse instead of a gaussian pi pulse in Rabi and change the total repetition times to 3000, you can do:

```python
quick.experiment.Rabi(var=v, data_path=DATA_PATH, g2_style="DRAG", g2_delta=-180, rep=3000).run()
```

- If you want to write your own pulse sequence or scan function, you can inherit or use the `quick.BaseExperiment`. You can put a Mercator protocol into `quick.experiment.configs`. This requires some advanced understanding of Mercator and Experiment layers. See API References for details.

- If you don't want to inspect how `quick.experiment` works, it might also be applicable to go one layer down. You can directly use Mercator *class*. With the help of `quick.Sweep` and `quick.Saver` in the Helpers, you can basically do everything you want.

- If you want to do some very fancy QICK board manipulation, you can always `import qick` and use the original qick firmware. `quick.Sweep` and `quick.Saver` will also be helpful.
