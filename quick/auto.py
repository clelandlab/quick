import quick.experiment as experiment
import quick.helper as helper
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import convolve1d, median_filter
from scipy.signal import find_peaks, peak_widths
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time

relevant_var = {
    "BaseAuto": [],
    "Resonator": ["r_freq", "r_power"],
    "QubitFreq": ["q_freq"],
    "PiPulseLength": ["q_length"],
    "PiPulseFreq": ["q_freq"],
    "ReadoutFreq": ["r_freq"],
    "ReadoutState": ["r_threshold", "r_phase"],
    "Ramsey": ["q_freq"]
}

class BaseAuto:
    var = {}
    data = None
    def __init__(self, var, silent=False, data_path=None, soccfg=None, soc=None):
        self.var = dict(var)
        self.silent = silent
        self.data_path = data_path
        self.soccfg = soccfg
        self.soc = soc
    def load_data(self, *paths):
        self.data = helper.load_data(*paths).T
    def update(self, v):
        for k in relevant_var[self.__class__.__name__]:
            v[k] = self.var[k]

class Resonator(BaseAuto):
    def calibrate(self):
        self.var["r_relax"] = 0
        if self.data is None:
            self.data = experiment.ResonatorSpectroscopy(data_path=self.data_path, title=f'(auto.Resonator) {int(self.var["r_freq"])}', r_power=np.arange(-60, -15, 1), r_freq=np.linspace(self.var["r_freq"] - 2, self.var["r_freq"] + 2, 100), soccfg=self.soccfg, soc=self.soc, var=self.var).run(silent=self.silent).data.T
        P, F, A = self.data[0], self.data[1], self.data[2]
        Fn = len(np.unique(F))
        F, P = F[0:Fn], P[0::Fn]
        unit = (F[-1] - F[0]) / Fn
        A = A.reshape((-1, Fn))
        avg = np.mean(A, axis=0)
        avg = convolve1d(avg, np.ones(3) / 3)
        peaks, _ = find_peaks(-avg, distance=0.3/unit, width=0.05/unit, prominence=0.3)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].plot(F, avg)
        axes[0].vlines(peaks * unit + F[0], ymin=np.min(avg), ymax=np.max(avg), color="red")
        axes[0].set_xlabel("Frequency (MHz)")
        axes[0].set_ylabel("Amplitude (dB) [log mag]")
        axes[0].grid()
        fi = peaks[-1]
        s = convolve1d(A[:, fi], np.ones(5) / 5)
        axes[1].plot(P, s, marker="o")
        axes[1].grid()
        axes[1].set_xlabel("Power (dBm)")
        if s[-1] - np.min(s) < np.std(A):
            return False, fig
        threshold = (np.max(s) - np.min(s)) * 0.2 + np.min(s)
        for i in range(len(s) - 1, 1, -1):
            if s[i] < threshold and (s[i] - s[i-1]) / (P[i] - P[i-1]) < 0.1:
                pi = i
                break
        fi = np.argmin(convolve1d(A[pi, :], np.ones(3) / 3))
        axes[1].vlines(P[pi], ymin=np.min(s), ymax=np.max(s), color="red")
        self.var["r_freq"] = float(F[fi])
        self.var["r_power"] = float(P[pi])
        return self.var, fig

class QubitFreq(BaseAuto):
    def calibrate(self, q_freq_min=3000, q_freq_max=5000, gain=0.5):
        self.var["r_relax"] = 1
        self.var["q_gain"] = gain
        def scan(scan_freq, title=""):
            self.data = experiment.QubitSpectroscopy(data_path=self.data_path, title=f'(auto.QubitFreq) {int(self.var["r_freq"])} {title}', q_freq=scan_freq, soccfg=self.soccfg, soc=self.soc, var=self.var).run(silent=self.silent).data.T
        if self.data is None: 
            scan(np.arange(q_freq_min, q_freq_max, 1))
        fig, ax = plt.subplots()
        F, A = self.data[0], self.data[1]
        Fn = len(F)
        unit = (F[-1]-F[0])/Fn
        ax.scatter(F, A, color="black", label="raw data", s=20)
        std = np.std(A)
        med = np.median(A)
        A = median_filter(A, size = 7)
        ax.plot(F, A, color="red", label="median filtered")
        ax.legend()
        ax.set_xlabel("Qubit Frequency (MHz)")
        ax.set_ylabel("Amplitude [lin mag]")
        ax.set_title("QubitSpectroscopy gain=0.5")
        ax.hlines([med + 2*std, med - 2*std], xmin=F[0], xmax=F[-1], color="green")
        ax.grid()
        peak, _ = find_peaks(A, height=med+2*std, distance=10/unit)
        print(F[peak])
        if len(peak) == 0:
            return False, fig
        if len(peak) == 1:
            self.var["q_freq"] = float(F[peak[0]])
            return self.var, fig
        width = peak_widths(A, peak, rel_height = 0.8)
        ls = []
        for w in range(len(width[0])):
            ls += [[F[peak[w]] - width[0][w]*unit, F[peak[w]] + width[0][w]*unit]]
        print(ls)
        for gain in np.arange(0.05, 0.4, 0.1):
            interval_score = []
            interval_peak = []
            self.var["q_gain"] = gain
            for i, l in enumerate(ls):
                scan(np.arange(l[0], l[1], 0.5), title=f"({i})gain={gain}")
                _A = median_filter(self.data[1], size=3)
                interval_score.append(np.max(_A) - np.min(_A))
                interval_peak.append(np.argmax(_A) * 0.5 + l[0])
            _score = np.sort(interval_score)
            if _score[-1] > 2 * _score[-2]:
                li = np.argmax(interval_score)
                print(interval_peak[li])
                self.var["q_freq"] = float(interval_peak[li])
                return self.var, fig
        return False, fig

class PiPulseLength(BaseAuto):
    def calibrate(self, q_length_max=0.5):
        def scan(scan_length, cycle=0):
            self.data = experiment.Rabi(soccfg=self.soccfg, soc=self.soc, var=self.var, data_path=self.data_path, title=f"(auto.PiPulseLength) {int(self.var['r_freq'])} length cycle={cycle}", q_length=scan_length).run(silent=self.silent).data.T
        if self.data is None:
            scan(np.arange(0.01, q_length_max, 0.01))
        fig, ax = plt.subplots()
        return False, fig

class PiPulseFreq(BaseAuto):
    pass

class ReadoutFreq(BaseAuto):
    def calibrate(self):
        if self.data is None:
            self.data = experiment.DispersiveSpectroscopy(soccfg=self.soccfg, soc=self.soc, var=self.var, data_path=self.data_path, title=f"(auto.ReadoutFreq) {int(self.var['r_freq'])}", r_freq=np.arange(v["r_freq"] - 1, v["r_freq"] + 1, 0.02)).run(silent=self.silent).data.T
        F, A0, P0, A1, P1 = data[0], data[1], data[2], data[5], data[6]
        dP = np.convolve(np.unwrap(P0 - P1), [0.333, 0.333, 0.333], "same")
        self.var["r_freq"] = float(F[np.argmax(dP)])
        fig, ax = plt.subplots()
        ax.plot(F, A0, label="Amplitude 0", marker="o")
        ax.plot(F, A1, label="Amplitude 1", marker="o")
        ax.vlines(self.var["r_freq"], ymin=np.min(A0), ymax=np.max(A0), colors="red", label="Max Phase Diff.")
        ax.legend()
        ax.set_xlabel("Readout Frequency (MHz)")
        ax.set_ylabel("Amplitude [log mag] (dB)")
        ax.set_title("DispersiveSpectroscopy")
        return self.var, fig

class ReadoutState(BaseAuto):
    pass

class Ramsey(BaseAuto):
    def calibrate(self, fringe_freq=10, max_time=1):
        self.var["fringe_freq"] = fringe_freq
        if self.data is None:
            self.data = experiment.T2Ramsey(soccfg=self.soccfg, soc=self.soc, var=self.var, data_path=self.data_path, title=f"(auto.Ramsey) {int(self.var['r_freq'])}", time=np.linspace(0, max_time, 100)).run(silent=self.silent).data.T
        L, A = self.data[0], self.data[1]
        def m(x, p1, p2, p3):
            return p1 * np.cos(p2 * x) + p3
        popt, pcov = curve_fit(m, L, A, p0=[(np.max(A) - np.min(A)) / 2, helper.estimateOmega(L, A), 0.5])
        fig, ax = plt.subplots()
        ax.scatter(L, A, s=10, color="black")
        ax.plot(np.arange(L[0], L[-1], 0.001), m(np.arange(L[0], L[-1], 0.001), *popt), color="red")
        ax.set_title(f"Ramsey (fringe_freq = {self.var['fringe_freq']})")
        self.var["q_freq"] = float(self.var["q_freq"] + popt[1] / 2 / np.pi - self.var["fringe_freq"])
        return self.var, fig

class Readout(BaseAuto):
    pass

def run(path, soccfg=None, soc=None, data_path=None):
    config = helper.load_yaml(path)
    qubits = config["qubits"]
    steps = config["steps"]
    qi = -1 # find the qubit to be run
    min_run = 9e9
    for i, q in enumerate(qubits):
        if q["status"].get("step", "start") in ["end", "fail"]:
            continue  # completed
        run = q["status"].get("run", 0)
        if run < min_run:
            qi = i
            min_run = run
    config["current"] = qi
    config["time"] = int(time.time() * 1000)
    helper.save_yaml(path, config)
    if qi < 0: # all completed
        return False
    step = qubits[qi]["status"].get("step", "start")
    print("\n------------ quick.auto.run ------------\n")
    print(f"qubits[{qi}]: {step}")
    skip = False
    try:
        a = globals()[steps[step].get("class", step)](var=qubits[qi]["var"], soccfg=soccfg, soc=soc, data_path=data_path)
    except:
        skip = True
        v = True
    try:
        if skip is False:
            v, fig = a.calibrate(**qubits[qi]["argument"].get(step, {}), **steps[step].get("argument", {}))
    except KeyboardInterrupt:
        print("\n! KeyboardInterrupt !")
        _config = helper.load_yaml(path) # avoid overwrite
        _config["current"] = -2
        _config["time"] = int(time.time() * 1000)
        helper.save_yaml(path, _config)
        return
    except:
        v = False
    qubits[qi]["status"]["run"] = qubits[qi]["status"].get("run", 0) + 1
    qubits[qi]["status"][step] = qubits[qi]["status"].get(step, 0) + 1
    if v is False: # failed
        if qubits[qi]["status"][step] >= 3:
            qubits[qi]["status"]["step"] = "fail"
        else:
            qubits[qi]["status"]["step"] = steps[step].get("fail", "fail")
    else: # succeeded
        if skip is False:
            a.update(qubits[qi]["var"])
        qubits[qi]["status"]["step"] = steps[step].get("next", "end")
    _config = helper.load_yaml(path) # avoid overwrite
    _config["current"] = -2
    _config["time"] = int(time.time() * 1000)
    _config["qubits"][qi] = qubits[qi]
    helper.save_yaml(path, _config)
    return True
