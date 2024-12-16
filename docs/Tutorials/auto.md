# Auto

This is a tutorial of `quick.auto`. For detailed parameters information, see API References.

## Task Config

Here is an example of task config file.

```yaml
current: -1
qubits:
- argument:
    PiPulseLength0:
      q_length_max: 1
    PiPulseLength1:
      q_length_max: 1
    QubitFreq:
      q_freq_max: 5500
      q_freq_min: 3000
  status:
    Resonator: 2
    run: 2
    step: fail
  var:
    q: 10
    r: 8
    r_freq: 6156.520202020202
    r_length: 4
    r_offset: 0.5
    r_relax: 500
steps:
  PiPulseFreq0:
    argument:
      p1_style: flat_top
      q_length: 2
      r: 20
    class: PiPulseFreq
    next: PiPulseLength0
  PiPulseFreq1:
    argument:
      cycles:
      - 1
      r: 10
    class: PiPulseFreq
    next: PiPulseLength1
  PiPulseLength0:
    argument:
      tol: 0.6
    back: fail
    back1: PiPulseFreq0
    back2: QubitFreq
    class: PiPulseLength
    next: PiPulseFreq1
  PiPulseLength1:
    argument:
      cycles:
      - 1
      - 5
      tol: 0.3
    back: fail
    back1: PiPulseFreq0
    class: PiPulseLength
    next: ReadoutFreq1
  QubitFreq:
    back: fail
    back1: Resonator
    next: PiPulseLength0
  Ramsey:
    class: Ramsey
  Readout:
    next: ReadoutFreq2
  ReadoutFreq1:
    class: ReadoutFreq
    next: ReadoutState1
  ReadoutFreq2:
    class: ReadoutFreq
    next: ReadoutState2
  ReadoutState1:
    back: fail
    class: ReadoutState
    next: Relax
  ReadoutState2:
    argument:
      tol: 0.5
    back: fail
    back1: Readout
    class: ReadoutState
    next: Ramsey
  Relax:
    next: Readout
  Resonator:
    back: fail
    back1: Resonator
    next: QubitFreq
  start:
    next: Resonator
time: 1
```

## Run Task

After connect to your QICK board (using `quick.connect`),

```python
quick.auto.run("task.yml")
```

This will run **one step** according to the task config file, and it will **modify** the content of the task config file to save the status and variables.

To automatically run through the task, use a `while` loop:

```python
while quick.auto.run("task.yml"):
    pass
    # clear_output() # clear output in Jupyter to avoid super long output
    # plt.show() # plot the figure from the auto steps.
```