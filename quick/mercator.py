from qick import NDAveragerProgram
from qick.averager_program import QickSweep
from .helper import dbm2gain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

listTypes = (list, np.ndarray)

def us2t(self, us, **kwargs): # a better us2cycles function, takes care of sweeping array
    return [self.us2cycles(us=us[0],  **kwargs), self.us2cycles(us=us[1],  **kwargs), us[2]] if isinstance(us, listTypes) else self.us2cycles(us=us, **kwargs)

def parse(self):
    """
    Creates the internal control object which controls the pulse sequence.
    c = {
        "g": {} # Generator channels
        "r": {} # Readout channels
        "exec": [] # Execution list
        "sw": {} # Sweep dictionary
    }
    """
    cfg = self.cfg
    steps = cfg.pop("steps", []) # flatten steps
    for i in range(len(steps)):
        for key in steps[i]:
            if not f"{i}_{key}" in cfg:
                cfg[f"{i}_{key}"] = steps[i][key]
    self.c = c = { "g": {}, "r": {}, "exec": [], "sw": {} } # internal control object
    cfg["reps"] = cfg.get("hard_avg", 1)
    cfg["soft_avgs"] = cfg.get("soft_avg", 1)
    def sweep(g, o, i=0):
        for k in ["freq", "phase", "gain", "t"]:
            if k in o and isinstance(o[k], listTypes):
                sw_key = i if k == "t" else f"g{g}_{k}"
                c["sw"][sw_key] = { "value": o[k], "key": k, "g": g }
                o[k] = o[k][0]
    def generate_waveform(o):
        g, l = o["g"], o["length"] * 16
        f_fabric = self.soccfg["gens"][g]["f_fabric"]
        samps_per_clk = self.soccfg["gens"][g]["samps_per_clk"]
        σ = us2t(self, o["sigma"] * samps_per_clk, gen_ch=g)
        if o["style"] == "const":
            o["idata"] = None
        if o["style"] == "flat_top":
            l = us2t(self, o["sigma"], gen_ch=g) * 5 * samps_per_clk
        if o["style"] in ["gaussian", "flat_top", "DRAG"]:
            x = np.arange(0, l)
            μ = l/2 - 0.5
            o["idata"] = 32766 * np.exp(-(x - μ) ** 2 / σ ** 2)
        if o["style"] == "DRAG":
            δ = -o["delta"] / (samps_per_clk * f_fabric)
            o["qdata"] = 0.5 * (x - μ) / (2 * σ ** 2) * o["idata"] / δ
    for g in range(10): # find all generator channels
        if f"g{g}_freq" in cfg:
            c["g"][g] = o = { "g": g, "r": None } # creates a single control object w/ any and all parameters
            o["freq"] = cfg.get(f"g{g}_freq")
            o["nqz"] = cfg.get(f"g{g}_nqz", 2)
            o["balun"] = cfg.get(f"g{g}_balun", 2)
            o["mode"] = cfg.get(f"g{g}_mode", "oneshot")
            o["style"] = cfg.get(f"g{g}_style", "const")
            o["phase"] = cfg.get(f"g{g}_phase", 0)
            o["length"] = cfg.get(f"g{g}_length", 2)
            o["sigma"] = cfg.get(f"g{g}_sigma", o["length"] / 5)
            o["delta"] = cfg.get(f"g{g}_delta", -200)
            o["length"] = us2t(self, o["length"], gen_ch=g) # us to ticks
            o["gain"] = cfg.get(f"g{g}_gain", 0)
            o["idata"] = cfg.get(f"g{g}_idata", None)
            o["qdata"] = cfg.get(f"g{g}_qdata", None)
            if f"g{g}_power" in cfg:
                o["gain"] = dbm2gain(cfg[f"g{g}_power"], o["freq"], o["nqz"], o["balun"])
            cfg[f"g{g}_gain"] = o["gain"] # write back computed gain
            generate_waveform(o)
            sweep(g, o)
    for r in range(3): # find all readout channels
        if f"r{r}_g" in cfg or f"r{r}_freq" in cfg:
            c["r"][r] = o = {}
            o["g"] = cfg.get(f"r{r}_g", None)
            o["freq"] = cfg.get(f"r{r}_freq", 0)
            o["length"] = us2t(self, cfg.get(f"r{r}_length", 2), ro_ch=r)
            if o["g"] is not None: # match corresponding g
                c["g"][o["g"]]["r"] = r
                o["freq"] = c["g"][o["g"]]["freq"]
    if cfg.get("rep", 0) > 0: # sweep for rep
        sweep(next(iter(c["g"])), { "t": [0, 0, cfg["rep"]] })
    for i in range(99999999): # find all execution steps
        if not f"{i}_type" in cfg:
            break
        o = { "type": cfg[f"{i}_type"] }
        o["ch"] = cfg.get(f"{i}_ch")
        o["t"] = cfg.get(f"{i}_t", 0)
        o["rep"] = cfg.get(f"{i}_rep", 1)
        if f"{i}_time" in cfg:
            o["t"] = us2t(self, cfg[f"{i}_time"])
        if o["type"] == "sync":
            sweep(next(iter(c["g"])), o, i)
        if o["type"] == "trigger" and o["ch"] is None:
            o["ch"] = c["r"].keys()
        if o["type"] == "set":
            o["key"] = cfg.get(f"{i}_key", "")
            o["value"] = cfg.get(f"{i}_value", 0)
            o["operator"] = cfg.get(f"{i}_operator", None)
        if o["type"] == "goto":
            o["i"] = cfg.get(f"{i}_i", i + 1)
        if o["type"] == "cond_pulse":
            o["threshold"] = cfg.get(f"{i}_threshold", 0)
        c["exec"].append(o)

class Mercator(NDAveragerProgram):
    """General class for preparing and sending a pulse sequence."""
    def initialize(self):
        parse(self)
        c = self.c
        for g, o in c["g"].items(): # Declare Generator Channels
            self.declare_gen(ch=g, nqz=o["nqz"])
        for r, o in c["r"].items(): # Declare Readout Channels
            self.declare_readout(ch=r, length=o["length"], freq=o["freq"], gen_ch=o["g"])
        for g, o in c["g"].items(): # convert values to reg
            o["freq"] = self.freq2reg(o["freq"], gen_ch=g, ro_ch=o["r"])
            o["phase"] = self.deg2reg(o["phase"], gen_ch=g)
        for _, sw in c["sw"].items(): # register sweeps
            if sw["key"] == "t":
                sw["reg"] = self.new_gen_reg(sw["g"], init_val=sw["value"][0], tproc_reg=True)
            else:
                sw["reg"] = self.get_gen_reg(sw["g"], sw["key"])
            self.add_sweep(QickSweep(self, sw["reg"], sw["value"][0], sw["value"][1], sw["value"][2]))            
        for g, o in c["g"].items(): # Setup pulse in each generator channel
            self.default_pulse_registers(ch=g, freq=o["freq"], phase=o["phase"], gain=o["gain"])
            if o["style"] == "const":
                self.set_pulse_registers(ch=g, style="const", length=o["length"], mode=o["mode"])
            elif o["style"] == "flat_top":
                self.add_pulse(ch=g, name=f"g{g}", idata=o["idata"], qdata=o["qdata"])
                self.set_pulse_registers(ch=g, style="flat_top", waveform=f"g{g}", length=o["length"])
            else:
                self.add_pulse(ch=g, name=f"g{g}", idata=o["idata"], qdata=o["qdata"])
                self.set_pulse_registers(ch=g, style="arb", waveform=f"g{g}", mode=o["mode"])
        self.synci(200)
    
    def body(self):
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
                self.pulse(ch=o["ch"], t=o["t"])
            if o["type"] == "wait_all":
                self.wait_all(t=o["t"])
            if o["type"] == "sync":
                if i in self.c["sw"]: # sweep
                    self.sync(self.c["sw"][i]["reg"].page, self.c["sw"][i]["reg"].addr)
                else:
                    self.synci(o["t"])
            if o["type"] == "sync_all":
                self.sync_all(t=o["t"])
            if o["type"] == "trigger":
                self.trigger(adcs=o["ch"], t=o["t"], pins=[0], adc_trig_offset=0)
            if o["type"] == "set":
                reg = self.get_gen_reg(o["ch"], o["key"])
                if o["operator"] is None:
                    reg.set_to(o["value"])
                else:
                    reg.set_to(reg, o["operator"], o["value"], physical_unit=(o["operator"] != "*"))
            if o["type"] == "goto":
                for j in range(o["i"], i):
                    reps[j] += self.c["exec"][j]["rep"]
                i = o["i"]
            if o["type"] == "cond_pulse":
                self.read(0, 0, "lower", 2)
                self.read(0, 0, "upper", 3)
                self.regwi(0, 6, int(o["threshold"] * self.c["r"][0]["length"]))
                self.condj(0, 2, "<", 6, f"cond_pulse_{i}")
                self.pulse(ch=o["ch"], t=o["t"])
                self.label(f"cond_pulse_{i}")

    def acquire_decimated(self, soc, progress=False): # Overwrites the default to disable the progress bar
        return super().acquire_decimated(soc, soft_avgs=self.cfg.get("soft_avgs", 1), progress=progress)

    def light(self):
        plt.clf()
        c = self.c
        us = self.cycles2us(1)
        us_r = self.cycles2us(1, ro_ch=0)
        data = {} # plot data
        for g in range(7, -1, -1):
            if g in c["g"]:
                data[g] = [[0, 0]]
        syncs = [0]
        sync = 0 # all t are absolute
        pulse_until = 0
        generator_until = np.zeros(9)
        def add_pulse(g, start, first=True):
            o = c["g"][g]
            if first:
                data[g].append([start, 0])
            if o["idata"] is not None:
                l = o["length"] if o["style"] == "flat_top" else 0
                end = start + l + len(o["idata"]) / 16
                h = len(o["idata"]) // 2
                data[g].extend(list(zip(np.arange(h)/16 + start, o["gain"] * np.array(o["idata"])[:h] / 32766)))
                data[g].extend(list(zip(np.arange(h)/16 + start + l + h / 16, o["gain"] * np.array(o["idata"])[h:] / 32766)))
            else:
                end = start + o["length"]
                data[g].extend([[start, o["gain"]], [end, o["gain"]]])
            if o["mode"] != "periodic":
                data[g].append([end, 0])
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
            start = sync + o["t"]
            end = start
            if o["type"] == "pulse":
                start = max(start, generator_until[o["ch"]])
                end = add_pulse(o["ch"], start)
                generator_until[o["ch"]] = max(generator_until[o["ch"]], end)
                pulse_until = max(pulse_until, end)
            if o["type"] == "sync":
                sync = sync + o["t"]
                syncs.append(sync)
            if o["type"] == "wait_all":
                end = max(start, pulse_until)
            if o["type"] == "sync_all":
                sync = max(pulse_until, sync) + o["t"]
                syncs.append(sync)
            if o["type"] == "set":
                c["g"][o["ch"]][o["key"]] = o["value"]
            if o["type"] == "goto":
                for j in range(o["i"], i):
                    reps[j] += self.c["exec"][j]["rep"]
                i = o["i"]
            if o["type"] == "trigger":
                for r in o["ch"]:
                    end = start + c["r"][r]["length"]
                    pulse_until = max(end, pulse_until)
                    plt.gca().add_patch(patches.Rectangle((start * us, -35000), (end - start) * us_r, 70000, fill=True, color=('r' if r == 0 else 'b'), alpha=0.1, label=f"r{r}"))
        final = max(pulse_until, sync)
        for g in data:
            if c["g"][g]["mode"] == "periodic":
                end = data[g][-1][0]
                while end < final:
                    end = add_pulse(g, end + 1, False)
            else:
                data[g].append([final, 0])
            xy = np.transpose(data[g])
            plt.plot(xy[0] * us, xy[1], label=f"g{g}", linewidth=3)
        plt.scatter(np.array(syncs) * us, np.zeros(len(syncs)) - 34500, label="sync", marker="^")
        plt.legend()
        plt.title("Mercator Light")
        plt.ylabel("Gain")
        plt.ylim([-35000, 35000])
        plt.xlabel("Time (μs)")
        plt.grid()
        plt.show()
