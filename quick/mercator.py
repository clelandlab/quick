from qick.asm_v2 import AveragerProgramV2, QickSweep1D, AsmV2
from .helper import dbm2gain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_waveform(o, soccfg):
    f_fabric = soccfg["gens"][o["g"]]["f_fabric"]
    samps_per_clk = soccfg["gens"][o["g"]]["samps_per_clk"]
    l = int(o["length"] * f_fabric) * samps_per_clk
    σ = o["sigma"] * samps_per_clk * f_fabric # keep float!
    if o["style"] == "const":
        o["idata"] = None
    if o["style"] == "flat_top":
        l = int(σ * 5 / samps_per_clk / 2) * 2 * samps_per_clk
    if o["style"] in ["gaussian", "flat_top", "DRAG"]:
        x = np.arange(0, l)
        μ = l/2 - 0.5
        o["idata"] = np.exp(-(x - μ) ** 2 / σ ** 2)
    if o["style"] == "DRAG":
        δ = -o["delta"] / (samps_per_clk * f_fabric)
        o["qdata"] = 0.5 * (x - μ) / (2 * σ ** 2) * o["idata"] / δ

def parse(soccfg, cfg):
    """
    Creates the internal control object which controls the pulse sequence.
    c = {
        "p": {} # pulses
        "g": {} # Generator channels
        "r": {} # Readout channels
        "exec": [] # Execution list
    }
    """
    c = { "p": {}, "g": {}, "r": {}, "exec": [] }
    steps = cfg.pop("steps", []) # flatten steps
    for i in range(len(steps)):
        for key in steps[i]:
            if not f"{i}_{key}" in cfg:
                cfg[f"{i}_{key}"] = steps[i][key]
    for i in range(99999999): # find all execution steps
        if not f"{i}_type" in cfg:
            break
        o = { "type": cfg[f"{i}_type"] }
        o["t"] = cfg.get(f"{i}_t", 0)
        o["rep"] = cfg.get(f"{i}_rep", 1)
        if o["type"] == "pulse":
            o["g"], o["p"], o["r"] = cfg.get(f"{i}_g"), cfg.get(f"{i}_p"), cfg.get(f"{i}_r", 0)
            c["g"][o["g"]] = { "p": o["p"], "r": None }
            c["p"][o["p"]] = { "g": o["g"], "r": None }
            o["threshold"] = cfg.get(f"{i}_threshold")
        if o["type"] == "trigger":
            o["rs"], o["p"] = cfg.get(f"{i}_rs"), cfg.get(f"{i}_p")
            if o["rs"] != None:
                for r in o["rs"]:
                    c["r"][r] = {}
        if o["type"] == "goto":
            o["i"] = cfg.get(f"{i}_i", i + 1)
        c["exec"].append(o)
    for p in c["p"]: # find all used pulses
        o = c["p"][p]
        c["g"][o["g"]]["nqz"] = cfg.get(f"p{p}_nqz", 2)
        o["freq"] = cfg.get(f"p{p}_freq", 0)
        c["g"][o["g"]]["freq"] = o["freq"]
        o["mode"] = cfg.get(f"p{p}_mode", "oneshot")
        o["style"] = cfg.get(f"p{p}_style", "const")
        o["phase"] = cfg.get(f"p{p}_phase", 0)
        o["length"] = cfg.get(f"p{p}_length", 2)
        o["sigma"] = cfg.get(f"p{p}_sigma", o["length"] / 5)
        o["delta"] = cfg.get(f"p{p}_delta", -200)
        o["gain"] = cfg.get(f"p{p}_gain", 0)
        o["idata"] = cfg.get(f"p{p}_idata", None)
        o["qdata"] = cfg.get(f"p{p}_qdata", None)
        if f"p{p}_power" in cfg:
            balun = cfg.get(f"p{p}_balun", 2)
            o["gain"] = dbm2gain(cfg[f"p{p}_power"], o["freq"], c["g"][o["g"]]["nqz"], balun)
            cfg[f"p{p}_gain"] = o["gain"] # write back computed gain
        generate_waveform(o, soccfg)
    for r in range(10): # find all readout channels
        if f"r{r}_p" in cfg or f"r{r}_freq" in cfg:
            c["r"][r] = o = { "g": None }
            o["p"] = cfg.get(f"r{r}_p", None)
            o["freq"] = cfg.get(f"r{r}_freq", 0)
            o["phase"] = cfg.get(f"r{r}_phase", 0)
            o["length"] = cfg.get(f"r{r}_length", 2)
            if o["p"] is not None: # match corresponding pulse/channels
                o["g"] = c["p"][o["p"]]["g"]
                o["freq"] = c["p"][o["p"]]["freq"]
                c["g"][o["g"]]["r"] = r
                c["p"][o["p"]]["r"] = r
    cfg["rep"] = cfg.get("rep", 0)
    return c

class Mercator(AveragerProgramV2):
    """General class for preparing and sending a pulse sequence."""
    def _initialize(self, cfg):
        c = self.c
        for g, o in c["g"].items(): # Declare Generator Channels
            kwargs = {}
            if self.soccfg["gens"][g]["has_mixer"]:
                kwargs["mixer_freq"] = o["freq"]
                kwargs["ro_ch"] = o["r"]
            self.declare_gen(ch=g, nqz=o["nqz"], **kwargs)
        for r, o in c["r"].items(): # Declare Readout Channels
            self.declare_readout(ch=r, length=o["length"])
            self.add_readoutconfig(ch=r, name=f"r{r}", freq=o['freq'], gen_ch=o["g"], phase=o["phase"])
            self.send_readoutconfig(ch=r, name=f"r{r}", t=0)
        for p, o in c["p"].items(): # Setup pulses
            kwargs = { "style": o["style"] }
            if o["style"] == "const":
                kwargs["mode"] = o["mode"]
            else: # non-const pulse
                kwargs["envelope"] = f"e{p}"
                maxv = self.soccfg.get_maxv(o["g"])
                idata = maxv * o["idata"] if o["idata"] is not None else None
                qdata = maxv * o["qdata"] if o["qdata"] is not None else None
                self.add_envelope(ch=o["g"], name=kwargs["envelope"], idata=idata, qdata=qdata)
            if o["style"] in ["flat_top", "const"]:
                kwargs["length"] = o["length"]
            else:
                kwargs["style"] = "arb"
            self.add_pulse(ch=o["g"], name=f"p{p}", ro_ch=o["r"], freq=o["freq"], phase=o["phase"], gain=o["gain"], **kwargs)
        if cfg["rep"] > 0:
            self.add_loop("rep", cfg["rep"])
        self.delay(0.5)
    
    def _body(self, cfg):
        reps = [] # get rep for each step
        for o in self.c["exec"]:
            reps.append(o["rep"])
        i = 0
        while i < len(reps):
            if reps[i] <= 0:
                i += 1
                continue
            reps[i] -= 1
            o = self.c["exec"][i]
            if o["type"] == "pulse":
                if o["threshold"] is not None: # conditional
                    self.read_and_jump(ro_ch=o["r"], component="I", threshold=self.us2cycles(o["threshold"] * self.c["r"][o["r"]]["length"], ro_ch=o["r"]), test="<", label=f"after_{i}")
                self.pulse(ch=o["g"], name=f"p{o['p']}", t=o["t"])
                if o["threshold"] is not None:
                    self.label(f"after_{i}")
            time_functions = { "delay": self.delay, "delay_auto": self.delay_auto, "wait": self.wait, "wait_auto": self.wait_auto }
            if o["type"] in time_functions:
                time_functions[o["type"]](o["t"])
            if o["type"] == "trigger":
                self.trigger(ros=(o["rs"] or self.c["r"].keys()), t=o["t"], pins=[0])
            if o["type"] == "goto":
                for j in range(o["i"], i):
                    reps[j] += self.c["exec"][j]["rep"]
                i = o["i"]
            
    def __init__(self, soccfg, cfg):
        self.c = parse(soccfg, cfg)
        super().__init__(soccfg, reps=cfg.get("hard_avg", 1), final_delay=0, cfg=cfg)

    def acquire(self, soc, progress=False, **kwargs):
        res = super().acquire(soc, progress=progress, soft_avgs=self.cfg.get("soft_avg", 1), **kwargs)
        return np.moveaxis(res, -1, 0)

    def acquire_decimated(self, soc, progress=False, **kwargs):
        res = super().acquire_decimated(soc, progress=progress, soft_avgs=self.cfg.get("soft_avg", 1), **kwargs)
        return np.moveaxis(res, -1, 0)

    def light(self):
        fig, ax = plt.subplots()
        c = self.c
        data = {} # plot data
        for g in range(15, -1, -1):
            if g in c["g"]:
                data[g] = [[0, 0]]
        delays = [0]
        delay = 0 # all t are absolute
        pulse_until = 0
        generator_until = np.zeros(17)
        periodic = {}
        def add_pulse(p, g, start, first=True):
            us = self.cycles2us(1, gen_ch=g) / self.soccfg["gens"][g]["samps_per_clk"]
            o = c["p"][p]
            if first:
                data[g].append([start, 0])
                ax.annotate(f"p{p}", (start, o["gain"] + 0.05))
            if o["idata"] is not None:
                l = o["length"] if o["style"] == "flat_top" else 0
                end = start + l + len(o["idata"]) * us
                h = len(o["idata"]) // 2
                data[g].extend(list(zip(np.arange(h)*us + start, o["gain"] * np.array(o["idata"])[:h])))
                data[g].extend(list(zip(np.arange(h)*us + start + l + h*us, o["gain"] * np.array(o["idata"])[h:])))
            else:
                end = start + o["length"]
                data[g].extend([[start, o["gain"]], [end, o["gain"]]])
            if o["mode"] != "periodic":
                data[g].append([end, 0])
            periodic[g] = (o["mode"] == "periodic")
            return end
        reps = [] # get rep for each step
        for o in self.c["exec"]:
            reps.append(o["rep"])
        i = 0
        while i < len(reps):
            if reps[i] <= 0:
                i += 1
                continue
            reps[i] -= 1
            o = self.c["exec"][i]
            start = delay + o["t"]
            end = start
            if o["type"] == "pulse":
                start = max(start, generator_until[o["g"]])
                end = add_pulse(o["p"], o["g"], start)
                generator_until[o["g"]] = max(generator_until[o["g"]], end)
                pulse_until = max(pulse_until, end)
            if o["type"] == "delay":
                delay = delay + o["t"]
                delays.append(delay)
            if o["type"] == "wait_auto":
                end = max(start, pulse_until)
            if o["type"] == "delay_auto":
                delay = max(pulse_until, delay) + o["t"]
                delays.append(delay)
            if o["type"] == "goto":
                for j in range(o["i"], i):
                    reps[j] += self.c["exec"][j]["rep"]
                i = o["i"]
            if o["type"] == "trigger":
                for r in (o["rs"] or c["r"]):
                    end = start + c["r"][r]["length"]
                    pulse_until = max(end, pulse_until)
                    ax.add_patch(patches.Rectangle((start, -1.05), end - start, 2.25, fill=True, color=('r' if r == 0 else 'b'), alpha=0.1, label=f"r{r}"))
        final = max(pulse_until, delay)
        for g in data:
            if periodic.get(g):
                end = data[g][-1][0]
                while end < final:
                    end = add_pulse(g, end + 1, False)
            else:
                data[g].append([final, 0])
            xy = np.transpose(data[g])
            ax.plot(xy[0], xy[1], label=f"g{g}", linewidth=3)
        ax.scatter(np.array(delays), np.zeros(len(delays)) - 1, label="delay", marker="^")
        ax.legend()
        ax.set_title("Mercator Light")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Gain")
        ax.set_ylim([-1.05, 1.2])
        ax.grid()
        return fig
