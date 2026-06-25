# Build an Experiment

This is a very simple tutorial to show how to build a customized experiment without using `quick.experiment`. For pre-defined experiments, see [Experiment](../experiment).

## 1. Connect to the QICK Board

> You need to install qick on your QICK board first. See [QICK Setup](../qick) for instructions.

Once your QICK board is set up and running, connect to it from the experiment computer in the same network:

```python
import quick, yaml
import numpy as np

soccfg, soc = quick.connect(QICK_IP)
print(soccfg)
```

It would be helpful to identify the relevant generator(DAC) and readout(ADC) channels for your experiment from `soccfg`.

## 2. Write Pulse Sequence

> The pulse sequence is specified by [Mercator Protocol](../mercator), see detailed tutorial documentation for full description.

In short, **Mercator Protocol** includes 4 parts:

1. Meta Information: specify the averaging and repetition.
2. Pulse Setup: define the pulses.
3. Readout Setup: specify the readout parameters.
4. Execution Steps: lay down the timeline of the pulse sequence.

**Example**:

We typically use YAML format to write the Mercator Protocol. For detailed description of all possible keys, see [Mercator Protocol](../mercator))

**1. Meta Information**: hard averaging 1000 times (run the sequence 1000 times on board and return the averaged result)

```yaml
hard_avg: 1000
```

**2. Pulse Setup**: define a constant(square) pulse `1` with frequency 5000 MHz, length 2 us, and power -30 dB (relative to the maximum output power of the generator):

```yaml
p1_style: const
p1_freq: 5000
p1_length: 2
p1_power: -30
```

In this section, all keys start with `p` followed by a pulse index. The pulse index can be arbitrary integers or letters.

**3. Readout Setup**: define a readout acquisition window with length 3 us, on readout(ADC) channel `0`:

```yaml
r0_p: 1 # links the pulse 1 for frequency matching
r0_length: 3
```

In this section, all keys start with `r` followed by the readout(ADC) channel index. The readout channel index can be found in `soccfg`. We need to specify the downconversion frequency for the readout channel. Here we link it to pulse `1` for frequency matching.

**4. Execution Steps**: now we can lay down the prepared things on a timeline. Here we first apply the pulse `1` on generator(DAC) channel `8`, and 0.1 us later trigger the readout(ADC) channel `0`. Then we wait for acquisition to finish and wait for another 2 us before the next repetition.

```yaml
steps:
- type: pulse
  p: 1
  g: 8
- type: trigger
  t: 0.1
- type: delay_auto
  t: 2
```

This section uses a list of steps contained in the `steps` key. All possible step types and keys are listed in [Mercator Protocol](../mercator).

## 3. Plot and Run the Pulse Sequence

Collect all the above parts into a single string and use the `quick.Mercator` class:

```python
mercator_protocol = """
hard_avg: 1000
p1_style: const
p1_freq: 5000
p1_length: 2
p1_power: -30
r0_p: 1
r0_length: 3
steps:
- type: pulse
  p: 1
  g: 8
- type: trigger
  t: 0.1
- type: delay_auto
  t: 2
"""

cfg = yaml.safe_load(mercator_protocol) # convert the string to a dictionary
m = quick.Mercator(soccfg, cfg) # initialize the Mercator class with the pulse sequence

# visualize the pulse sequence
m.light()

# execute the pulse sequence
I, Q = m.acquire(soc) # return averaged values of I and Q from the readout channel
```

> Detailed description about arguments and return values of `quick.Mercator` can be found in [Mercator API](../../API References/mercator).

## 4. Variable Insertion

Using a fixed pulse sequence is not very flexible. Besides programmatically modifying the dictionary `cfg`, there is a cooler way to apply variables to the pulse sequence. For example, we want to use a dictionary `v` to specify the pulse frequency(`v["r_freq"]`) and readout(ADC) channel(`v["rr"]`):

```python
v = { "r_freq": 5000, "rr": 0 } # example

mercator_protocol = """
hard_avg: 1000
p1_style: const
p1_freq: {r_freq}   # look here
p1_length: 2
p1_power: -30
r{rr}_p: 1          # and here
r{rr}_length: 3     #
steps:
- type: pulse
  p: 1
  g: 8
- type: trigger
  t: 0.1
- type: delay_auto
  t: 2
"""

# before converting the string to a dictionary,
# use `quick.evalStr` to insert the variables into the string
cfg = yaml.safe_load(quick.evalStr(mercator_protocol, v))
```

This results in the same `cfg` dictionary as before, but now we can easily change the pulse frequency and readout channel by modifying the dictionary `v` without touching the Mercator Protocol string.

> See [evalStr API](../../API References/helper/#evalstr) for more details

You can also directly modify the dictionary `cfg` after it is created:

```python
cfg["hard_avg"] = 2000 # change hard averaging to 2000
```

## 5. Sweeping Parameters

What if we want to run a pulse sequence with different pulse frequencies? We can use `quick.Sweep` to sweep the variables:

```python
sweep = { "r_freq": np.arange(5000, 6000, 1) }
for _v in quick.Sweep(v, sweep):
    # here _v is the variables with sweeping r_freq
    cfg = yaml.safe_load(quick.evalStr(mercator_protocol, _v))
    m = quick.Mercator(soccfg, cfg)
    I, Q = m.acquire(soc)
```

You can also directly sweep the dictionary `cfg` after it is created:

```python
sweep = { "p1_power": np.arange(-30, -20, 1) }
for _cfg in quick.Sweep(cfg, sweep):
    m = quick.Mercator(soccfg, _cfg)
    I, Q = m.acquire(soc)
```

> See [Sweep API](../../API References/helper/#sweep) for more details

## 6. Save Data

Don't forget to save the data! You can use `quick.Saver` to save the data to files:

```python
# initialize the Saver
s = quick.Saver("Example Title", "path/to/directry",
    indep_params=[("Frequency", "MHz")],
    dep_params=[("Amplitude", "a.u.")]
    )

sweep = { "r_freq": np.arange(5000, 6000, 1) }
for _v in quick.Sweep(v, sweep):
    cfg = yaml.safe_load(quick.evalStr(mercator_protocol, _v))
    m = quick.Mercator(soccfg, cfg)
    I, Q = m.acquire(soc)
    S = I[0][0] + 1j * Q[0][0]
    s.write_data([ # append rows of data
        [ _v["r_freq", np.abs(S) ]
    ])

# record the finishing time
s.write_yml()
```

> See [Saver API](../../API References/helper/#saver) for more details

