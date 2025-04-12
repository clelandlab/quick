import quick.experiment as experiment
import quick.helper as helper
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import convolve1d, median_filter
from scipy.signal import find_peaks, peak_widths
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time, os

relevant_var = {
    "BaseAuto": [],
    "ReadoutLength": ["r_length"],
    "Resonator": ["r_freq", "r_power"],
    "QubitFreq": ["q_freq"],
    "PiPulseLength": ["q_length"],
    "PiPulseFreq": ["q_freq"],
    "ReadoutFreq": ["r_freq"],
    "ReadoutState": ["r_threshold", "r_phase"],
    "Relax": ["r_relax"],
    "Readout": ["r_power", "r_length"],
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

class ReadoutLength(BaseAuto):
    def calibrate(self, **kwargs):
        self.var["r_relax"] = 0
        self.var["r_power"] = 0
        def scan(label, f, w):
            self.data = experiment.ResonatorSpectroscopy(
                data_path=self.data_path, title=f'(auto.ReadoutLength) {int(self.var["r_freq"])} {label}',
                r_freq=np.linspace(f - w, f + w, 1001),
                p0_mode="periodic", r0_length=213, hard_avg=10, # VNA style
                soccfg=self.soccfg, soc=self.soc, var=self.var, **kwargs
            ).run(silent=self.silent).data.T
        if self.data is None:
            scan("wide", self.var["r_freq"], 5)
        data = self.data
        p, perr, r2, fig = helper.fitResonator(data[0], data[3] + 1j * data[4], fit="circle")
        if r2 < 0.5 or perr[1] > 0.2 * p[1]:
            experiment.LoopBack(soccfg=self.soccfg, soc=self.soc, var=self.var).run(silent=True)
            return False, fig
        f, Qc = p[2], p[1]
        last_data = self.data
        scan("focus", f, f / Qc / 2)
        experiment.LoopBack(soccfg=self.soccfg, soc=self.soc, var=self.var).run(silent=True)
        data = np.hstack([last_data, self.data])
        p, perr, r2, fig = helper.fitResonator(data[0], data[3] + 1j * data[4], fit="circle")
        if r2 < 0.5 or perr[1] > 0.2 * p[1]:
            return False, fig
        f, Qc = p[2], p[1]
        self.var["r_length"] = float(min(Qc / f, 10))
        return self.var, fig

class Resonator(BaseAuto):
    def calibrate(self, **kwargs):
        self.var["r_relax"] = 0
        if self.data is None:
            self.data = experiment.ResonatorSpectroscopy(
                data_path=self.data_path, title=f'(auto.Resonator) {int(self.var["r_freq"])}',
                r_power=np.arange(-50, 0, 2), r_freq=np.linspace(self.var["r_freq"] - 2, self.var["r_freq"] + 2, 101),
                p0_mode="periodic", r0_length=213, hard_avg=10, # VNA style
                soccfg=self.soccfg, soc=self.soc, var=self.var, **kwargs
            ).run(silent=self.silent).data.T
            experiment.LoopBack(soccfg=self.soccfg, soc=self.soc, var=self.var).run(silent=True)
        P, F, A = self.data[0], self.data[1], self.data[2]
        Fn = len(np.unique(F))
        F, P = F[0:Fn], P[0::Fn]
        unit = (F[-1] - F[0]) / Fn
        A = A.reshape((-1, Fn))
        avg = np.mean(A, axis=0)
        avg = convolve1d(avg, np.ones(3) / 3)
        peaks, _ = find_peaks(-avg, distance=0.3/unit, width=0.05/unit, prominence=0.3, height=-((np.max(avg) - np.min(avg)) * 0.7 + np.min(avg)))
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].plot(F, avg)
        axes[0].vlines(peaks * unit + F[0], ymin=np.min(avg), ymax=np.max(avg), color="red")
        axes[0].set_xlabel("Frequency (MHz)")
        axes[0].set_ylabel("Amplitude (dB)")
        axes[0].grid()
        fi = peaks[-1]
        s = convolve1d(A[:, fi], np.ones(5) / 5)
        axes[1].plot(P, s, marker="o")
        axes[1].grid()
        axes[1].set_xlabel("Power (dB)")
        if s[-1] - np.min(s) < np.std(A):
            return False, fig
        threshold = (np.max(s) - np.min(s)) * 0.4 + np.min(s)
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
    def calibrate(self, q_freq_min=3000, q_freq_max=5000, q_gain=0.5, **kwargs):
        self.var["r_relax"] = 1
        self.var["q_gain"] = q_gain
        self.var["q_length"] = 2
        def scan(scan_freq, title=""):
            self.data = experiment.QubitSpectroscopy(data_path=self.data_path, title=f'(auto.QubitFreq) {int(self.var["r_freq"])} {title}', q_freq=scan_freq, soccfg=self.soccfg, soc=self.soc, var=self.var, **kwargs).run(silent=self.silent).data.T
        if self.data is None: 
            scan(np.arange(q_freq_min, q_freq_max, 1))
        fig, ax = plt.subplots()
        F, A = self.data[0], self.data[1]
        Fn = len(F)
        unit = (F[-1]-F[0])/Fn
        ax.scatter(F, A, color="black", label="raw data", s=20)
        std = np.std(A)
        med = np.median(A)
        A = median_filter(A, size = 5)
        ax.plot(F, A, color="red", label="median filtered")
        ax.legend()
        ax.set_xlabel("Qubit Frequency (MHz)")
        ax.set_ylabel("Amplitude [lin mag]")
        ax.set_title("QubitSpectroscopy gain=0.5")
        ax.hlines([med + 2*std, med - 2*std], xmin=F[0], xmax=F[-1], color="green")
        ax.grid()
        peak, _ = find_peaks(A, height=med+2*std, distance=10/unit)
        ax.scatter(F[peak], A[peak], marker="x", color="red")
        print("Peaks: ", F[peak])
        if len(peak) == 0:
            return False, fig
        if len(peak) == 1:
            self.var["q_freq"] = float(F[peak[0]])
            return self.var, fig
        width = peak_widths(A, peak, rel_height=0.7)
        ls = []
        for w in range(len(width[0])):
            ls += [[F[peak[w]] - width[0][w]*unit, F[peak[w]] + width[0][w]*unit]]
        for gain in np.arange(0.05, 0.4, 0.1):
            interval_score = []
            interval_peak = []
            self.var["q_gain"] = gain
            for i, l in enumerate(ls):
                scan(np.arange(l[0], l[1], 0.5), title=f"({i})gain={round(gain, 2)}")
                _A = median_filter(self.data[1], size=3)
                interval_score.append(max(np.max(_A) - med - 1.5*std, 0) / std)
                interval_peak.append(np.argmax(_A) * 0.5 + l[0])
            print("Interval Score: ", interval_score)
            _score = np.sort(interval_score)
            if _score[-1] > _score[-2] + 0.2:
                li = np.argmax(interval_score)
                print(interval_peak[li])
                self.var["q_freq"] = float(interval_peak[li])
                return self.var, fig
        return False, fig

class PiPulseLength(BaseAuto):
    def calibrate(self, q_length_max=0.5, cycles=[], tol=0.5, **kwargs):
        fig, axes = plt.subplots(len(cycles) + 1, 1)
        def scan(scan_length, cycle=0):
            self.data = experiment.Rabi(soccfg=self.soccfg, soc=self.soc, var=self.var, data_path=self.data_path, title=f"(auto.PiPulseLength) {int(self.var['r_freq'])} cycle={cycle}", q_length=scan_length, cycle=cycle, rep=2000, **kwargs).run(silent=self.silent).data.T
        def fit(cycle=0, ax=axes):
            L, A = self.data[0], self.data[2]
            def m(x, p1, p2, p3):
                return p1 * np.sin(x * p2) ** 2 + p3
            popt, pcov = curve_fit(m, L, A, p0=[np.max(A) - np.min(A), helper.estimateOmega(L, A) / 2, np.min(A)], bounds=([0.1, 0.1, 0.1], [np.inf, np.inf, np.inf]))
            T = np.pi / popt[1]
            residuals = A - m(L, *popt)
            dof = len(L) - len(popt)
            r2 = 1 - np.sum((residuals - np.mean(residuals))**2) / np.sum((A - np.mean(A))**2)
            ax.scatter(L, A, color="black", s=20)
            ax.plot(L, m(L, *popt), color="blue")
            ax.set_xlabel("Pi Pulse Length (us)")
            self.var["q_length"] = float(T * (cycle / 2 + 0.5))
            ax.vlines([self.var["q_length"]], ymin=np.min(A), ymax=np.max(A), color="red")
            print("R^2 =", r2)
            return r2
        if self.data is None:
            scan(np.linspace(0.008, q_length_max, 100))
        r2 = fit(cycle=0, ax=(axes[0] if len(cycles) > 0 else axes))
        if r2 < tol:
            return False, fig
        for j in range(len(cycles)):
            c = cycles[j]
            if c <= 0:
                continue
            scan(np.linspace(self.var["q_length"] * (c - 0.5) / (c + 0.5), self.var["q_length"] * (c + 1.5) / (c + 0.5), 51), cycle=c)
            r2 = fit(cycle=c, ax=axes[j+1])
            if r2 < tol:
                return False, fig
        return self.var, fig

class PiPulseFreq(BaseAuto):
    def calibrate(self, cycles=[], r=10, **kwargs):
        fig, axes = plt.subplots(len(cycles) + 1, 1)
        def scan(cycle=0):
            self.data = experiment.Rabi(soccfg=self.soccfg, soc=self.soc, var=self.var, data_path=self.data_path, title=f"(auto.PiPulseFreq) {int(self.var['r_freq'])} cycle={cycle}", q_freq=np.linspace(self.var["q_freq"]-r,self.var["q_freq"]+r,101), cycle=cycle, rep=2000, **kwargs).run(silent=self.silent).data.T
        def fit(ax=axes):
            F, A = self.data[0], self.data[2]
            self.var["q_freq"] = float(helper.symmetryCenter(F, A))
            ax.plot(F, A, marker="o")
            ax.set_xlabel("Pi Pulse Freq (MHz)")
            ax.vlines([self.var["q_freq"]], ymin=np.min(A), ymax=np.max(A), color="red")
        if self.data is None:
            scan(cycle=0)
        fit(ax=(axes[0] if len(cycles) > 0 else axes))
        for j in range(len(cycles)):
            c = cycles[j]
            if c <= 0:
                continue
            scan(cycle=c)
            fit(ax=axes[j+1])
        return self.var, fig

class ReadoutFreq(BaseAuto):
    def calibrate(self, r=1, **kwargs):
        if self.data is None:
            self.data = experiment.DispersiveSpectroscopy(soccfg=self.soccfg, soc=self.soc, var=self.var, data_path=self.data_path, title=f"(auto.ReadoutFreq) {int(self.var['r_freq'])}", r_freq=np.linspace(self.var["r_freq"] - r, self.var["r_freq"] + r, 100), **kwargs).run(silent=self.silent).data.T
        F, A0, P0, I0, Q0, A1, P1, I1, Q1 = self.data
        dS_amp = np.convolve(np.abs(I0 + 1j*Q0 - I1 - 1j*Q1), np.ones(5) / 5, "same")
        self.var["r_freq"] = float(F[np.argmax(dS_amp)])
        fig, ax = plt.subplots()
        ax.plot(F, A0, label="Amplitude 0 (dB)", marker="o")
        ax.plot(F, A1, label="Amplitude 1 (dB)", marker="o")
        axr = ax.twinx() 
        axr.set_ylabel('Separation') 
        axr.plot(F, dS_amp, label="Separation", color="green")
        ax.vlines([self.var["r_freq"]], ymin=np.min(A0), ymax=np.max(A0), colors="red", label="Max Separation")
        ax.legend()
        ax.set_xlabel("Readout Frequency (MHz)")
        ax.set_ylabel("Amplitude (dB)")
        ax.set_title("DispersiveSpectroscopy")
        return self.var, fig

class ReadoutState(BaseAuto):
    def calibrate(self, tol=0.1, **kwargs):
        self.var["r_phase"] = 0
        if self.data is None:
            self.data = experiment.IQScatter(var=self.var, soccfg=self.soccfg, soc=self.soc, data_path=self.data_path, title=f"(auto.ReadoutState)", **kwargs).run(silent=self.silent).data.T
        c0, c1, visibility, Fg, Fe, fig = helper.iq_scatter(self.data[0] + 1j * self.data[1], self.data[2] + 1j * self.data[3])
        if visibility < tol:
            return False, fig
        self.var["r_phase"], self.var["r_threshold"] = helper.iq_rotation(c0, c1)
        return self.var, fig 

class Ramsey(BaseAuto):
    def calibrate(self, fringe_freq=10, max_time=1, **kwargs):
        self.var["fringe_freq"] = fringe_freq
        if self.data is None:
            self.data = experiment.T2Ramsey(soccfg=self.soccfg, soc=self.soc, var=self.var, data_path=self.data_path, title=f"(auto.Ramsey) {int(self.var['r_freq'])}", time=np.linspace(0, max_time, 101), **kwargs).run(silent=self.silent).data.T
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
    def calibrate(self):
        self.var["r_length"] = self.var.get("r_length", experiment.var["r_length"])
        def negative_score(x):
            self.var["r_power"], self.var["r_length"] = float(x[0]), float(x[1])
            data = experiment.IQScatter(var=self.var, soccfg=self.soccfg, soc=self.soc).run(silent=True).data.T
            c0, c1, visibility, Fg, Fe, fig = helper.iq_scatter(data[0] + 1j * data[1], data[2] + 1j * data[3])
            c1m = np.median(data[2]) + 1j * np.median(data[3])
            c1_shift = np.abs(c1 - c1m) / np.abs(c1m - c0)
            plt.close()
            score = visibility - abs(Fg - Fe) - 10 * c1_shift ** 2
            print(x, score)
            return -score
        initial_simplex = [[self.var["r_power"], self.var["r_length"]], [self.var["r_power"] + 4, self.var["r_length"]], [self.var["r_power"], self.var["r_length"] + 2]]
        bounds = [(-60, 0), (0.1, 10)]
        minimize(negative_score, x0=[self.var["r_power"], self.var["r_length"]], bounds=bounds, method="Nelder-Mead", options={ "maxfev": 100, "xatol": 0.05, "fatol": 1, "initial_simplex": initial_simplex })
        return self.var, None

class Relax(BaseAuto):
    def calibrate(self, **kwargs):
        if self.data is None:
            self.data = experiment.T1(var=self.var, data_path=self.data_path, soccfg=self.soccfg, soc=self.soc, title=f"(auto.Relax) {int(self.var['r_freq'])}", time=np.linspace(0, self.var["r_relax"] * 0.8, 61), **kwargs).run(silent=self.silent).data.T
        popt, _, _, fig = helper.fitT1(self.data[0], self.data[1])
        print("T1 =", popt[1])
        self.var["r_relax"] = float(5 * popt[1])
        return self.var, fig

def run(path, soccfg=None, soc=None, data_path=None):
    config = helper.load_yaml(path)
    qubits = config["qubits"]
    steps = config["steps"]
    qi = -1 # find the qubit to be run
    min_run = 9e9
    for i, q in enumerate(qubits):
        if q["status"].get("step", "start") in ["end", "fail", "pause"]:
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
    print(f"\n------------ quick.auto.run [{qi}] {step} ------------\n")
    skip = False
    try:
        a = globals()[steps[step].get("class", step)](var=qubits[qi]["var"], soccfg=soccfg, soc=soc, data_path=data_path)
    except:
        skip = True
        v = True
    try:
        if skip is False:
            v, fig = a.calibrate(**steps[step].get("argument", {}), **qubits[qi]["argument"].get(step, {}))
    except KeyboardInterrupt:
        print("\n!!!!! KeyboardInterrupt !!!!!")
        _config = helper.load_yaml(path) # avoid overwrite
        _config["current"] = -2
        _config["time"] = int(time.time() * 1000)
        helper.save_yaml(path, _config)
        return
    except Exception as e:
        print(e)
        v = False
    qubits[qi]["status"]["run"] = qubits[qi]["status"].get("run", 0) + 1
    qubits[qi]["status"][step] = qubits[qi]["status"].get(step, 0) + 1
    i = qubits[qi]["status"][step]
    if v is False: # failed
        qubits[qi]["status"]["step"] = steps[step].get(f"back{i}", steps[step].get("back", step))
    else: # succeeded
        if skip is False:
            a.update(qubits[qi]["var"])
        qubits[qi]["status"]["step"] = steps[step].get(f"next{i}", steps[step].get("next", "end"))
    _config = helper.load_yaml(path) # avoid overwrite
    _config["current"] = -2
    _config["time"] = int(time.time() * 1000)
    _config["qubits"][qi] = qubits[qi]
    helper.save_yaml(path, _config)
    return True

