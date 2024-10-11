# Mercator Protocol

Mercator protocol is a syntax to specify a pulse sequence (QICK program). It is a set of key-value pairs. Conveniently, it can be writen as a Python dictionary or in [YAML format](https://en.wikipedia.org/wiki/YAML). This document will use YAML format for clearance.

> More examples of Mercator protocol can be found in [default experiment programs](https://github.com/clelandlab/quick/blob/main/quick/constants/experiment.yml). Note they are up to variable insertion by `quick.evalStr`.

Mercator protocol includes a lot of default values and syntax sugar to make it simultaneously flexible and easy to use. In this document, **default values will be used. Properties without default value will be marked as required.**

Mercator protocol is generally consisted of 5 sections:

- Meta information
- Generator channel setup
- Readout channel setup
- Execution steps

For example, the following program is for QubitSpectroscopy (TwoTone):

```yaml
# section 1: Meta Information
reps: 1000
# section 2: Generator Channel Setup
g0_freq: 5000
g0_length: 3
g0_power: -40
g0_balun: 3
g2_freq: 4000
g2_style: flat_top
g2_sigma: 0.05
g2_length: 1
g2_gain: 10000
# section 3: Readout Channel Setup
r0_g: 0
r0_length: 2
# section 4: Execution Steps
0_type: pulse
0_ch: 2
1_type: sync_all
2_type: pulse
2_ch: 0
3_type: trigger
3_time: 0.5
4_type: wait_all
5_type: sync_all
5_time: 500
```

## Meta Information

This section often describes the program repetition and average times. You can also include your own key-value pairs, as long as they are not conflicting with other keys.

```yaml
reps: 1            # on-board repetition (average) times
soft_avgs: 1       # average times in Python
```

## Generator Channel Setup

In this section, you can prepare your generator (DAC) channels. All properties in this section have prefix `gx_`, where `x` represents the generator channel number. This document use `g0_` as an example.

```yaml
g0_freq: 5000      # (REQUIRED) [MHz] frequency
g0_gain: 0         # [-32766, 32766] gain
g0_nqz: 2          # [1, 2]nyquist zone
g0_balun: 2        # [0, 3] balun
g0_mode: oneshot   # [oneshot|periodic]
g0_style: const    # [const|gaussian|DRAG|flat_top|arb] pulse style
g0_phase: 0        # [deg] phase
g0_length: 2       # [us] length
g0_delta: -200     # [MHz] anharmonicity used in DRAG pulse
g0_idata: []       # [-32766, 32766] used in arb pulse, sampling 16x clock ticks
g0_qdata: []       # [-32766, 32766] used in arb pulse, sampling 16x clock ticks
```

**Syntax Sugar**: (optional)

```yaml
g0_power: -40      # [dBm] power, will overwrite g0_gain
g0_sigma: 0.05     # [us] gaussian std in flat_top/gaussian/DRAG pulse.
                   # Its default value is 1/5 of g0_length
```

## Readout Channel Setup

In this section, you can prepare your readout (ADC) channels. All properties in this section have prefix `rx_`, where `x` represents the readout channel number. This document use `r0_` as an example.

```yaml
r0_freq: 0         # [MHz] readout frequency
r0_length: 2       # [us] readout length
```

**Syntax Sugar**: (recommended)

```yaml
r0_g: 0            # match one generator channel for frequency down-conversion
                   # will overwrite r0_freq
```

## Execution Steps

In this section, you can describe a series of steps to be executed during the run time. All properties in this section have prefix `i_` where `i` is the step number starting from `0`. All step numbers MUST be consecutive. The program will stop if the next consecutive number is not found. This document use step `0` as an example.

Each step will be in the following syntax: 

```yaml
0_type: pulse      # (REQUIRED) [pulse|trigger|wait_all|sync|sync_all|set|goto]
0_ch: 0            # (required for pulse, set) which generator channel pulse
0_time: 0          # [us] time offset from last sync
0_reps: 1          # repetition times of this step
```

The step `type` is required. It takes one of the following values:

- `pulse`: release a pulse from channel `0_ch`.
- `trigger`: trigger the readout. `0_ch` should be an array, defaulted to all readout channels.
- `wait_all`: wait for all pulses and readouts
- `sync_all`: sync channels after the end of all pulses and readouts (+`0_time`)
- `sync`: offset the channel sync by `0_time`
- `set`: (discussed below)
- `goto`: (discussed below)

**type: set**

`set` step can update register values on the fly. It takes the following syntax:

```yaml
0_type: set
0_ch: 0            # (REQUIRED) generator channel
0_key: phase       # (REQUIRED) name of the register
0_value: 0         # (REQUIRED) value to be set
0_operator: +      # [+|-|*] if provided, the register will be set incrementally.
0_time: 0          # [us] time offset from last sync
0_reps: 1          # repetition times of this step
```

**type: goto**

`goto` step can jump to another step. It takes the following syntax:

```yaml
0_type: goto
0_i: 0             # (REQUIRED) target step number
0_reps: 1          # repetition times of this step
```

> Note: if `goto` a previous step, everything in between will be executed again, respecting the `reps` of the steps. The `goto` function will NOT create actual loops in the assembly code. Instead, it will expand the loop in Python and generate a program without loops.

**Syntax Sugar**: (sometimes useful)

The execution steps can also be writen as an array in Mercator protocol, for example:

```yaml
steps:
  - type: pulse
    ch: 0
  - type: trigger
    time: 0.5
  - type: wait_all
  - type: sync_all
    time: 1
```

The Mercator *class* accepts any **combination** of the array format and the flat key-value format as previously discussed. But it will first **flatten** the `steps` array into the flat key-value format. **The flat key-value format has higher priority during this flattening process.**

## Other Features

**Sweeping** on board is supported! It is supported for generator on `gain`, `phase`, and `freq` (`freq` only supports **non-readout** pulse), and on execution type `sync` (for time sweeping). You can pass in sweep values like `[start, end, number of points]`. For example:

```yaml
g0_gain: [10000, 20000, 11]

4_type: sync
4_time: [0, 2, 51]
```

For each sweep, the return data will have an extra axis. The axis order is always in `(time, gain, phase, freq, time series)`, if existing. The maximum acquire size is **1024** data in total. Try to avoid sweeping multiple dimensions on board!
