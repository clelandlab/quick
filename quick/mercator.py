from qick.asm_v2 import AveragerProgramV2, QickSweep1D, AsmV2
from .helper import dB2gain
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_waveform(o, soccfg):
    f_fabric = soccfg["gens"][o["g"]]["f_fabric"]
    samps_per_clk = soccfg["gens"][o["g"]]["samps_per_clk"]
    l = int(o["length"] * f_fabric) * samps_per_clk
    σ = o["sigma"] * samps_per_clk * f_fabric # keep float!
    if o["style"] == "const":
        o["idata"] = None
    if o["style"] in ["flat_top", "stage"]: # l computed from σ
        l = int(σ * 5 / samps_per_clk / 2) * 2 * samps_per_clk
    if o["style"] in ["gaussian", "flat_top", "DRAG"]:
        x = np.arange(0, l)
        μ = l/2 - 0.5
        o["idata"] = np.exp(-(x - μ) ** 2 / σ ** 2)
    if o["style"] == "stage":
        o["idata"] = np.zeros(int(σ * 4))
        for s in o["stage"]:
            o["idata"] = np.append(o["idata"], np.zeros(int(s[1] * f_fabric * samps_per_clk)) + s[0])
        o["idata"] = np.append(o["idata"], np.zeros(int(σ*4)))
        # pad o["idata"] to be integer multiple of samps_per_clk
        o["idata"] = np.append(o["idata"], np.zeros(samps_per_clk - len(o["idata"]) % samps_per_clk))
        o["idata"] = gaussian_filter(o["idata"], σ)
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
            o["rep"] = cfg.get(f"{i}_rep", 0)
            o["i"] = cfg.get(f"{i}_i", i + 1)
        c["exec"].append(o)
    for p in c["p"]: # find all used pulses
        o = c["p"][p]
        c["g"][o["g"]]["nqz"] = cfg.get(f"p{p}_nqz", 2)
        o["freq"] = cfg.get(f"p{p}_freq", 0)
        c["g"][o["g"]]["freq"] = o["freq"]
        c["g"][o["g"]]["mixer"] = c["g"][o["g"]].get("mixer", cfg.get(f"p{p}_mixer", None))
        o["mode"] = cfg.get(f"p{p}_mode", "oneshot")
        o["phrst"] = cfg.get(f"p{p}_phrst", 0)
        o["style"] = cfg.get(f"p{p}_style", "const")
        o["length"] = cfg.get(f"p{p}_length", 2)
        o["sigma"] = cfg.get(f"p{p}_sigma", o["length"] / 5)
        o["delta"] = cfg.get(f"p{p}_delta", -200)
        o["gain"] = cfg.get(f"p{p}_gain", 0)
        o["mask"] = cfg.get(f"p{p}_mask", list(range(len(o["freq"]))) if np.iterable(o["freq"]) else None)
        o["phase"] = cfg.get(f"p{p}_phase", list(np.zeros(len(o["freq"]))) if np.iterable(o["freq"]) else 0)
        o["stage"] = cfg.get(f"p{p}_stage", [])
        o["idata"] = np.array(cfg[f"p{p}_idata"]) if cfg.get(f"p{p}_idata") is not None else None
        o["qdata"] = np.array(cfg[f"p{p}_qdata"]) if cfg.get(f"p{p}_qdata") is not None else None
        if cfg.get(f"p{p}_power", None) is not None:
            o["gain"] = dB2gain(cfg[f"p{p}_power"])
            cfg[f"p{p}_gain"] = o["gain"] # write back computed gain
        generate_waveform(o, soccfg)
    for r in range(10): # find all readout channels
        if f"r{r}_p" in cfg or f"r{r}_freq" in cfg:
            c["r"][r] = o = { "g": None }
            o["p"] = cfg.get(f"r{r}_p")
            o["freq"] = cfg.get(f"r{r}_freq")
            o["phase"] = cfg.get(f"r{r}_phase", 0)
            o["length"] = cfg.get(f"r{r}_length", 2)
            if o["p"] is not None: # match corresponding pulse/channels
                o["g"] = c["p"][o["p"]]["g"]
                c["g"][o["g"]]["r"] = c["p"][o["p"]]["r"] = r
                if o["freq"] is None:
                    o["freq"] = c["p"][o["p"]]["freq"]
    cfg["rep"] = cfg.get("rep", 0)
    return c

class Mercator(AveragerProgramV2):
    """General class for preparing and sending a pulse sequence."""
    def _initialize(self, cfg):
        c = self.c
        mux_gain_factor = { 1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8 }
        for g, o in c["g"].items(): # Declare Generator Channels
            kwargs = {}
            if self.soccfg["gens"][g]["has_mixer"]:
                kwargs["mixer_freq"] = o["mixer"] or int(o["freq"] / 100) * 100
                kwargs["ro_ch"] = o["r"]
                if np.iterable(o["freq"]): # mux
                    kwargs["mixer_freq"] = o["mixer"] or np.mean(o["freq"])
                    kwargs["mux_freqs"] = o["freq"]
                    kwargs["mux_gains"] = np.array(c["p"][o["p"]]["gain"]) * mux_gain_factor[len(o["freq"])]
                    kwargs["mux_phases"] = c["p"][o["p"]]["phase"]
            self.declare_gen(ch=g, nqz=o["nqz"], **kwargs)
        for r, o in c["r"].items(): # Declare Readout Channels
            kwargs = { "phase": o["phase"], "freq": o["freq"] }
            if o.get("g") is not None:
                kwargs["gen_ch"] = o["g"]
            if "tproc_ctrl" in self.soccfg["readouts"][r]:
                self.declare_readout(ch=r, length=o["length"])
                self.add_readoutconfig(ch=r, name=f"r{r}", **kwargs)
                self.send_readoutconfig(ch=r, name=f"r{r}", t=0)
            else: # mux readout
                self.declare_readout(ch=r, length=o["length"], **kwargs)
        for p, o in c["p"].items(): # Setup pulses
            kwargs = { "style": o["style"], "ro_ch": o["r"], "freq": o["freq"], "phase": o["phase"], "gain": o["gain"], "phrst": o["phrst"] }
            if o["style"] == "const":
                kwargs["mode"] = o["mode"]
                if o["mask"] is not None: # mux mask
                    kwargs["mask"] = o["mask"]
                    for k in ["freq", "ro_ch", "mode", "phase", "gain"]:
                        kwargs.pop(k, None)
            else: # non-const pulse
                kwargs["envelope"] = f"e{p}"
                maxv = self.soccfg.get_maxv(o["g"])
                idata = None if o["idata"] is None else maxv * np.array(o["idata"])
                qdata = None if o["qdata"] is None else maxv * np.array(o["qdata"])
                self.add_envelope(ch=o["g"], name=kwargs["envelope"], idata=idata, qdata=qdata)
            if o["style"] in ["flat_top", "const"]:
                kwargs["length"] = o["length"]
            else:
                kwargs["style"] = "arb"
            self.add_pulse(ch=o["g"], name=f"p{p}",  **kwargs)
        if cfg["rep"] > 0:
            self.add_loop("rep", cfg["rep"])
    
    def _body(self, cfg):
        c = self.c
        goto_rep = {} # goto rep
        i = 0
        while i < len(c["exec"]):
            o = c["exec"][i]
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
                goto_rep[i] = goto_rep.get(i, o["rep"]) - 1
                i = o["i"] if goto_rep[i] >= 0 else i + 1
            else:
                i = i + 1

    def __init__(self, soccfg, cfg):
        self.c = parse(soccfg, cfg)
        super().__init__(soccfg, reps=cfg.get("hard_avg", 1), final_delay=None, final_wait=None, cfg=cfg)

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
        delays = [0] # all t are absolute
        delay = pulse_until = 0
        generator_until = np.zeros(17)
        periodic = {}
        r_label = {}
        def add_pulse(p, g, start, first=True):
            us = self.cycles2us(1, gen_ch=g) / self.soccfg["gens"][g]["samps_per_clk"]
            o = c["p"][p]
            gain = o["gain"][0] if np.iterable(o["gain"]) else o["gain"]
            if first:
                data[g].append([start, 0])
                ax.annotate(f"p{p}", (start, gain + 0.05))
            if o["idata"] is not None:
                l = o["length"] if o["style"] == "flat_top" else 0
                end = start + l + len(o["idata"]) * us
                h = len(o["idata"]) // 2
                data[g].extend(list(zip(np.arange(h)*us + start, gain * np.array(o["idata"])[:h])))
                data[g].extend(list(zip(np.arange(h)*us + start + l + h*us, gain * np.array(o["idata"])[h:])))
            else:
                end = start + o["length"]
                data[g].extend([[start, gain], [end, gain]])
            if o["mode"] != "periodic":
                data[g].append([end, 0])
            periodic[g] = (o["mode"] == "periodic" and p)
            return end
        goto_rep = {}
        i = 0
        while i < len(c["exec"]):
            o = c["exec"][i]
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
            if o["type"] == "delay_auto":
                delay = max(pulse_until, delay) + o["t"]
                delays.append(delay)
            if o["type"] == "wait":
                for k in range(len(generator_until)):
                    generator_until[k] = max(generator_until[k], start)
            if o["type"] == "wait_auto":
                end = start = pulse_until + o["t"]
                for k in range(len(generator_until)):
                    generator_until[k] = max(generator_until[k], start)
            if o["type"] == "trigger":
                for r in (o["rs"] or c["r"]):
                    end = start + c["r"][r]["length"]
                    pulse_until = max(end, pulse_until)
                    ax.axvspan(xmin=start, xmax=end, color=('r' if r == 0 else 'b'), alpha=0.1, label=(None if r_label.get(r) else f"r{r}"))
                    r_label[r] = True
            if o["type"] == "goto":
                goto_rep[i] = goto_rep.get(i, o["rep"]) - 1
                i = o["i"] if goto_rep[i] >= 0 else i + 1
            else:
                i = i + 1
        final = max(pulse_until, delay)
        for g in data:
            if periodic.get(g) is not False:
                end = data[g][-1][0]
                while end < final:
                    end = add_pulse(periodic[g], g, end + 1, False)
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
