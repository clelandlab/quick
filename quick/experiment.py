import numpy as np
import quick.helper as helper
from quick import __version__
from .mercator import Mercator
import os, yaml

listTypes = (list, np.ndarray)

# global var & config for quick.experiment
configs = helper.load_yaml(os.path.join(os.path.abspath(os.path.dirname(__file__)), "./constants/experiment.yml"))
var = configs["var"]

class BaseExperiment:
    var = {}
    var_label = {}
    def __init__(self, data_path=None, title="", soccfg=None, soc=None, var=None, **kwargs):
        self.key = self.__class__.__name__ # get class name as experiment key
        self.data_path = data_path
        self.title = title
        template_var = dict(configs["var"])
        template_var.update(self.var)
        self.var = template_var
        self.var.update(var or {})
        template_var_label = dict(configs["var_label"])
        template_var_label.update(self.var_label)
        self.var_label = template_var_label
        self.sweep = {}
        self.config_update = {
            "quick.experiment": self.key,
            "quick.__version__": __version__
        }
        for k, v in kwargs.items():
            if k in self.var:
                self.var[k] = v
                if isinstance(v, listTypes):
                    self.sweep[k] = v
                    self.var[k] = v[0]
                if hasattr(self.var[k], "dtype"): # np to native for saving.
                    self.var[k] = self.var[k].item()
            else:
                self.config_update[k] = v
        self.config_update["var"] = self.var
        self.eval_config(self.var)
        self.soccfg, self.soc = helper.getSoc() if soccfg is None else (soccfg, soc)
        self.m = Mercator(self.soccfg, self.config)
    def eval_config(self, v):
        configStr = configs.get(self.key, "")
        self.config = yaml.safe_load(helper.evalStr(configStr, v)) or {}
        self.config.update(self.config_update)
    def prepare(self, indep_params=[], log_mag=False, population=False): # prepare saver
        self.data = []
        indep_params = list(indep_params) # avoid modify in place
        for k in self.sweep:
            label = self.var_label.get(k, (k, ""))
            indep_params.append(label)
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
    def acquire_S21(self, indep_list, log_mag=False, population=False): # acquire & add data
        self.m = Mercator(self.soccfg, self.config)
        I, Q = self.m.acquire(self.soc)
        S21 = (np.array(I) + 1j * np.array(Q)).flatten()
        dep_list = []
        if population:
            I = S21.real
            dep_list.append(float(I[I > self.var.get("r_threshold", 0)].size) / I.size)
        S21 = np.mean(S21)
        dep_list.extend([(20 * np.log10(np.abs(S21) / self.config["p0_gain"]) if log_mag else np.abs(S21)), np.angle(S21), S21.real, S21.imag])
        self.add_data([[*indep_list, *dep_list]])
    def run(self, silent=False, log_mag=False, population=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        self.prepare(log_mag=log_mag, population=population)
        for v in helper.Sweep(self.var, self.sweep, progressBar=(not silent)):
            self.eval_config(v)
            indep = []
            for k in self.sweep:
                indep.append(v[k])
            self.var = v
            self.acquire_S21(indep, log_mag=log_mag, population=population)
        return self.conclude(silent)
    def conclude(self, silent=False): # last step of run
        if not silent:
            print(f"quick.experiment({self.key}) Completed. Data saved in {self.data_path}" if self.data_path is not None else f"quick.experiment({self.key}) Completed.")
        self.data = np.array(self.data)
        return self
    def light(self):
        self.m.light()

# All experiments are following
class LoopBack(BaseExperiment):
    def run(self, silent=False, log_mag=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        self.prepare([("Time", "us")], log_mag=log_mag)
        self.eval_config(self.var)
        self.m = Mercator(self.soccfg, self.config)
        I, Q = self.m.acquire_decimated(self.soc, progress=not silent)
        t = self.m.get_time_axis(ro_index=self.var["rr"])
        S21 = I[0] + 1j * Q[0]
        self.add_data(np.transpose([t, (20 * np.log10(np.abs(S21) / self.config["p0_gain"]) if log_mag else np.abs(S21)), np.angle(S21), S21.real, S21.imag]))
        return self.conclude(silent)

class ResonatorSpectroscopy(BaseExperiment):
    def run(self, silent=False, log_mag=True):
        return super().run(silent, log_mag)

class QubitSpectroscopy(BaseExperiment):
    pass

class Rabi(BaseExperiment):
    def __init__(self, **kwargs):
        self.var = { "cycle": 0 } # add one var
        self.var_label = { "cycle": ("Extra Cycles", "") }
        super().__init__(**kwargs)
    def run(self, silent=False, population=True):
        return super().run(silent=silent, population=population)

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
        I1, Q1 = self.m.acquire(self.soc)
        I1, Q1 = I1[0][0], Q1[0][0]
        self.config["0_type"] = "delay_auto" # omit pi pulse
        self.m = Mercator(self.soccfg, self.config)
        I0, Q0 = self.m.acquire(self.soc)
        I0, Q0 = I0[0][0], Q0[0][0]
        self.add_data(np.transpose([I0, Q0, I1, Q1]))
        return self.conclude(silent)

class ActiveReset(BaseExperiment):
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        self.data = []
        indep_params = [("Qubit Pulse Gain", "a.u.")]
        dep_params = [("Population", "", "before reset"), ("Population", "", "after reset")]
        if self.data_path is not None:
            self.s = helper.Saver(f"({self.key})" + self.title, self.data_path, indep_params, dep_params, self.config)
        for c in helper.Sweep(self.config, { "p2_gain": np.arange(0, 1, 0.01) }, progressBar=(not silent)):
            self.m = Mercator(self.soccfg, c)
            I, Q = self.m.acquire(self.soc)
            I1, I2 = I[0]
            P1 = float(I1[I1 > self.var["r_threshold"]].size) / I1.size
            P2 = float(I2[I2 > self.var["r_threshold"]].size) / I2.size
            self.add_data([[c["p2_gain"], P1, P2 ]])
        return self.conclude(silent)

class DispersiveSpectroscopy(BaseExperiment):
    def __init__(self, r_freq=[], **kwargs):
        super().__init__(**kwargs)
        self.r_freq = r_freq
    def run(self, silent=False):
        if not silent:
            print(f"quick.experiment({self.key}) Starting")
        self.data = []
        indep_params = [("Readout Pulse Frequency", "MHz")]
        dep_params = [("Amplitude 0", "dB", "log mag"), ("Phase 0", "rad"), ("I 0", ""), ("Q 0", ""), ("Amplitude 1", "dB", "log mag"), ("Phase 1", "rad"), ("I 1", ""), ("Q 1", "")]
        if self.data_path is not None:
            self.s = helper.Saver(f"({self.key})" + self.title, self.data_path, indep_params, dep_params, self.config)
        for c in helper.Sweep(self.config, { "p0_freq": self.r_freq }, progressBar=(not silent)):
            c["0_type"] = "pulse" # send pi pulse
            self.m = Mercator(self.soccfg, c)
            I1, Q1 = self.m.acquire(self.soc)
            S1 = I1[0][0] + 1j * Q1[0][0]
            c["0_type"] = "delay_auto" # omit pi pulse
            self.m = Mercator(self.soccfg, c)
            I0, Q0 = self.m.acquire(self.soc)
            S0 = I0[0][0] + 1j * Q0[0][0]
            self.add_data([[c["p0_freq"], 20 * np.log10(np.abs(S0) / c["p0_gain"]), np.angle(S0), np.real(S0), np.imag(S0), 20 * np.log10(np.abs(S1) / c["p0_gain"]), np.angle(S1), np.real(S1), np.imag(S1) ]])
        return self.conclude(silent)

class T1(BaseExperiment):
    def __init__(self, **kwargs):
        self.var = { "time": 0 } # add one var
        self.var_label = { "time": ("Delay Time", "us") }
        super().__init__(**kwargs)
    def run(self, silent=False, population=True):
        return super().run(silent=silent, population=population)

class T2Ramsey(BaseExperiment):
    def __init__(self, **kwargs):
        self.var = { "time": 0, "fringe_freq": 0 } # add var
        self.var_label = {
            "time": ("Delay Time", "us"),
            "fringe_freq": ("Fringe Frequency", "MHz")
        }
        super().__init__(**kwargs)
    def run(self, silent=False, population=True):
        return super().run(silent=silent, population=population)

class T2Echo(BaseExperiment):
    def __init__(self, **kwargs):
        self.var = { "time": 0, "cycle": 0, "fringe_freq": 0 } # add var
        self.var_label = {
            "time": ("Delay Time", "us"),
            "cycle": ("Extra Cycles", ""),
            "fringe_freq": ("Fringe Frequency", "MHz")
        }
        super().__init__(**kwargs)
    def run(self, silent=False, population=True):
        return super().run(silent=silent, population=population)
