import numpy as np
import os, yaml
import quick.helper as helper
from quick import __version__
from .mercator import Mercator

# global var & config for quick.experiment
configs = helper.load_yaml(os.path.join(os.path.abspath(os.path.dirname(__file__)), "./constants/experiment.yml"))
var = configs["var"]

class BaseExperiment:
    """Experiment Base Class"""
    def __init__(self, data_path=None, title="", soccfg=None, soc=None, var=None, **kwargs):
        self.key = self.__class__.__name__ # get class name as experiment key
        self.data_path = data_path
        self.title = title
        self.var = dict(configs["var"])
        self.var.update(var or {})
        configStr = configs.get(self.key, "")
        self.config = yaml.safe_load(helper.evalStr(configStr, self.var)) or {}
        self.config["quick.experiment"] = self.key # label the experiment in config
        self.config["quick.__version__"] = __version__
        self.config["var"] = self.var
        self.config.update(kwargs) # customized arguments
        self.soccfg, self.soc = helper.getSoc() if soccfg is None else (soccfg, soc)
        self.m = Mercator(self.soccfg, self.config)
    def prepare(self, indep_params=[], log_mag=False, population=False): # prepare saver
        self.data = []
        dep_params = []
        if population:
            dep_params.append(("Population", ""))
        dep_params.extend([("Amplitude", "dB", "log mag") if log_mag else ("Amplitude", "", "lin mag"), ("Phase", "rad"), ("I", ""), ("Q", "")])
        if self.data_path is not None:
            self.s = helper.Saver(f"({self.key})" + self.title, self.data_path, indep_params, dep_params, self.config)
    def add_data(self, data): # add & save data
        self.data.extend(data)
        if self.data_path is not None:
            self.s.write_data(data)
    def acquire_S21(self, cfg, indep_list, log_mag=False, decimated=False, population=False): # acquire & add data
        self.m = Mercator(self.soccfg, cfg)
        iq_list = self.m.acquire_decimated(self.soc, progress=True) if decimated else self.m.acquire(self.soc)
        if decimated:
            iq = np.transpose(iq_list[0])
            S21 = np.array([iq[0] + 1j * iq[1]]).flatten()
        else:
            S21 = np.array([iq_list[1][0][0] + 1j * iq_list[2][0][0]]).flatten()
        dep_list = []
        if population:
            I = S21.real
            dep_list.append(float(I[I > self.var.get("r_threshold", 0)].size) / I.size)
        if not decimated:
            S21 = np.mean(S21)
        dep_list.extend([(20 * np.log10(np.abs(S21) / cfg["g0_gain"]) if log_mag else np.abs(S21)), np.angle(S21), S21.real, S21.imag])
        if decimated:
            self.add_data(np.transpose([*indep_list, *dep_list]))
        else:
            self.add_data([ [*indep_list, *dep_list] ])
    def conclude(self, silent=False): # last step of run
        if not silent:
            print(f"quick.experiment({self.key}) Completed. Data saved in {self.data_path}" if self.data_path is not None else f"quick.experiment({self.key}) Completed.")
        self.data = np.array(self.data)
        return self
    def light(self):
        self.m.light()

# All experiments are following
class LoopBack(BaseExperiment):
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        self.prepare([("Time", "us")])
        self.acquire_S21(self.config, [np.arange(self.soccfg.us2cycles(self.config["r0_length"], ro_ch=0)) * self.soccfg.cycles2us(1, ro_ch=0)], log_mag=False, decimated=True)
        return self.conclude(silent)

class ResonatorSpectroscopy(BaseExperiment):
    def __init__(self, r_freqs=[], r_powers=None, **kwargs):
        super().__init__(**kwargs)
        self.r_freqs = r_freqs
        self.r_powers = r_powers
    def run(self, silent=False, log_mag=True):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        hasPowers = self.r_powers is not None
        indep_params = [("Power", "dBm"), ("Frequency", "MHz")] if hasPowers else [("Frequency", "MHz")]
        sweep = { "g0_power": self.r_powers, "g0_freq": self.r_freqs } if hasPowers else { "g0_freq": self.r_freqs }
        self.prepare(indep_params, log_mag=log_mag)
        for c in helper.Sweep(self.config, sweep, progressBar=(not silent)):
            indep = [c["g0_power"], c["g0_freq"]] if hasPowers else [c["g0_freq"]]
            self.acquire_S21(c, indep, log_mag=log_mag)
        return self.conclude(silent)

class QubitSpectroscopy(BaseExperiment):
    def __init__(self, q_freqs=[], r_freqs=None, **kwargs):
        super().__init__(**kwargs)
        self.q_freqs = q_freqs
        self.r_freqs = r_freqs
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        indep_params = [("Qubit Frequency", "MHz")]
        sweep = { "g2_freq": self.q_freqs }
        if self.r_freqs is not None:
            indep_params.append(("Readout Frequency", "MHz"))
            sweep["g0_freq"] = self.r_freqs
        self.prepare(indep_params)
        for c in helper.Sweep(self.config, sweep, progressBar=(not silent)):
            indep = [c["g2_freq"]]
            if self.r_freqs is not None:
                indep.append(c["g0_freq"])
            self.acquire_S21(c, indep)
        return self.conclude(silent)

class Rabi(BaseExperiment):
    def __init__(self, q_lengths=None, q_gains=None, q_freqs=None, cycles=None, **kwargs):
        super().__init__(**kwargs)
        self.q_lengths = q_lengths
        self.q_gains = q_gains
        self.q_freqs = q_freqs
        self.cycles = cycles
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        indep_params = []
        sweep = {}
        if self.cycles is not None:
            indep_params.append(("Extra Cycles", ""))
            sweep["2_rep"] = self.cycles
        if self.q_lengths is not None:
            indep_params.append(("Pulse Length", "us"))
            sweep["g2_length"] = self.q_lengths
        if self.q_gains is not None:
            indep_params.append(("Pulse Gain", "a.u."))
            sweep["g2_gain"] = self.q_gains
        if self.q_freqs is not None:
            indep_params.append(("Qubit Frequency", "MHz"))
            sweep["g2_freq"] = self.q_freqs
        self.prepare(indep_params, population=True)
        for c in helper.Sweep(self.config, sweep, progressBar=(not silent)):
            indep = []
            if self.cycles is not None:
                indep.append(c["2_rep"])
                c["2_rep"] = 2 * c["2_rep"]
            if self.q_lengths is not None:
                indep.append(c["g2_length"])
            if self.q_gains is not None:
                indep.append(c["g2_gain"])
            if self.q_freqs is not None:
                indep.append(c["g2_freq"])
            self.acquire_S21(c, indep, population=True)
        return self.conclude(silent)

class IQScatter(BaseExperiment):
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        self.data = []
        indep_params = []
        dep_params = [("I 0", ""), ("Q 0", ""), ("I 1", ""), ("Q 1", "")]
        if self.data_path is not None:
            self.s = helper.Saver(f"({self.key})" + self.title, self.data_path, indep_params, dep_params, self.config)
        self.config["0_type"] = "pulse" # send pi pulse
        self.m = Mercator(self.soccfg, self.config)
        iq_list = self.m.acquire(self.soc)
        I1, Q1 = np.array(iq_list[1][0][0]), np.array(iq_list[2][0][0])
        self.config["0_type"] = "sync_all" # omit pi pulse
        self.m = Mercator(self.soccfg, self.config)
        iq_list = self.m.acquire(self.soc)
        I0, Q0 = np.array(iq_list[1][0][0]), np.array(iq_list[2][0][0])
        self.add_data(np.transpose([I0, Q0, I1, Q1]))
        return self.conclude(silent)

class DispersiveSpectroscopy(BaseExperiment):
    def __init__(self, r_freqs=[], **kwargs):
        super().__init__(**kwargs)
        self.r_freqs = r_freqs
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        self.data = []
        indep_params = [("Frequency", "MHz")]
        dep_params = [("Amplitude 0", "dB", "log mag"), ("Phase 0", "rad"), ("I 0", ""), ("Q 0", ""), ("Amplitude 1", "dB", "log mag"), ("Phase 1", "rad"), ("I 1", ""), ("Q 1", "")]
        if self.data_path is not None:
            self.s = helper.Saver(f"({self.key})" + self.title, self.data_path, indep_params, dep_params, self.config)
        for c in helper.Sweep(self.config, { "g0_freq": self.r_freqs }, progressBar=(not silent)):
            c["0_type"] = "pulse" # send pi pulse
            self.m = Mercator(self.soccfg, c)
            iq_list = self.m.acquire(self.soc)
            S1 = iq_list[1][0][0] + 1j * iq_list[2][0][0]
            c["0_type"] = "sync_all" # omit pi pulse
            self.m = Mercator(self.soccfg, c)
            iq_list = self.m.acquire(self.soc)
            S0 = iq_list[1][0][0] + 1j * iq_list[2][0][0]
            self.add_data([[c["g0_freq"], 20 * np.log10(np.abs(S0) / c["g0_gain"]), np.angle(S0), np.real(S0), np.imag(S0), 20 * np.log10(np.abs(S1) / c["g0_gain"]), np.angle(S1), np.real(S1), np.imag(S1) ]])
        return self.conclude(silent)

class T1(BaseExperiment):
    def __init__(self, times=[], **kwargs):
        super().__init__(**kwargs)
        self.times = times
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        indep_params = [("Pulse Delay", "us")]
        sweep = { "1_time": self.times }
        self.prepare(indep_params, population=True)
        for c in helper.Sweep(self.config, sweep, progressBar=(not silent)):
            indep = [c["1_time"]]
            self.acquire_S21(c, indep, population=True)
        return self.conclude(silent)

class T2Ramsey(BaseExperiment):
    def __init__(self, times=[], fringe_freqs=None, **kwargs):
        super().__init__(**kwargs)
        self.times = times
        self.fringe_freqs = fringe_freqs
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        indep_params = [("Pulse Delay", "us")]
        sweep = { "3_time": self.times }
        if self.fringe_freqs is not None:
            indep_params.append(("Fringe Frequency", "MHz"))
            sweep["fringe_freq"] = self.fringe_freqs
        self.prepare(indep_params, population=True)
        for c in helper.Sweep(self.config, sweep, progressBar=(not silent)):
            indep = [c["3_time"]]
            if self.fringe_freqs is not None:
                indep.append(c["fringe_freq"])
                self.var["fringe_freq"] = c["fringe_freq"]
            c["2_value"] = -360 * c["3_time"] * self.var["fringe_freq"] % 360
            self.acquire_S21(c, indep, population=True)
        return self.conclude(silent)

class T2Echo(BaseExperiment):
    def __init__(self, times=[], fringe_freqs=None, cycle=0, **kwargs):
        super().__init__(**kwargs)
        self.times = times
        self.fringe_freqs = fringe_freqs
        self.cycle = cycle
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        indep_params = [("Pulse Delay", "us")]
        N = self.cycle + 1
        sweep = { "4_time": np.array(self.times) / N / 2 }
        self.config["7_rep"] = self.cycle
        if self.fringe_freqs is not None:
            indep_params.append(("Fringe Frequency", "MHz"))
            sweep["fringe_freq"] = self.fringe_freqs
        self.prepare(indep_params, population=True)
        for c in helper.Sweep(self.config, sweep, progressBar=(not silent)):
            indep = [c["4_time"] * 2 * N] # total time
            if self.fringe_freqs is not None:
                indep.append(c["fringe_freq"])
                self.var["fringe_freq"] = c["fringe_freq"]
            c["6_time"] = c["4_time"] # half wait time
            c["8_value"] = -360 * 2 * N * c["4_time"] * self.var["fringe_freq"] % 360
            self.acquire_S21(c, indep, population=True)
        return self.conclude(silent)
