import quick.experiment as experiment
import quick.helper as helper
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import convolve1d, median_filter
from scipy.signal import find_peaks
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

relevant_var = {
    "BaseAuto": [],
    "Resonator": ["r_freq", "r_power"],
    "QubitFreq": ["q_freq"],
    "Ramsey": ["q_freq"]
}

class BaseAuto:
    var = {}
    data = [[]]
    def __init__(self, var, silent=False, data_path=None, soccfg=None, soc=None):
        self.var = dict(var)
        self.silent = silent
        self.data_path = data_path
        self.soccfg = soccfg,
        self.soc = soc
    def load_data(self, *paths):
        self.data = helper.load_data(*paths).T
    def update(self, v):
        for k in relevant_var[self.__class__.__name__]:
            v[k] = self.var[k]

class Resonator(BaseAuto):
    def calibrate(self):
        self.var["r_relax"] = 1
        if not self.data:
            self.data = experiment.ResonatorSpectroscopy(data_path=self.data_path, title=f'(auto.Resonator) {int(self.var["r_freq"])}', r_power=np.arange(-60, -15, 1), r_freq=np.linspace(self.var["r_freq"] - 2, self.var["r_freq"] + 2, 100), soccfg=self.soccfg, soc=self.soc, var=self.var).run(silent=self.silent).data.T
        P, F, A = self.data[0], self.data[1], self.data[2]
        Fn = len(np.unique(F))
        F, P = F[0:Fn], P[0::Fn]
        unit = (F[-1] - F[0]) / Fn
        A = A.reshape((-1, Fn))
        avg = np.mean(A, axis=0)
        avg = convolve1d(avg, np.ones(3) / 3)
        peaks, _ = find_peaks(-avg, distance=0.3/unit, width=0.05/unit, prominence=0.2)
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
        if s[-1] - s[0] < np.std(A):
            return False, fig
        threshold = (np.max(s) - np.min(s)) * 0.2 + np.min(s)
        for i in range(len(s) - 1, 0, -1):
            if s[i] < threshold:
                pi = i
                break
        fi = np.argmin(A[pi, :])
        axes[1].vlines(P[pi], ymin=np.min(s), ymax=np.max(s), color="red")
        self.var["r_freq"] = float(F[fi])
        self.var["r_power"] = float(P[pi])
        return self.var, fig

class QubitFreq(BaseAuto):
    def calibrate(self, q_freq=np.arange(3000, 4000, 1)):
        self.var["r_relax"] = 1
        if not self.data:
            self.data = experiment.QubitSpectroscopy(data_path=self.data_path, title=f'(auto.QubitFreq) {int(self.var["r_freq"])}', q_freq=q_freq, soccfg=self.soccfg, soc=self.soc, var=self.var).run(silent=self.silent).data.T
        # Todo.
        fig, ax = plt.subplots()
        return False, fig

class PiPulseLength(BaseAuto):
    pass

class PiPulseFreq(BaseAuto):
    pass

class ReadoutFreq(BaseAuto):
    def calibrate(self):
        if not self.data:
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

class Ramsey(BaseAuto):
    def calibrate(self, fringe_freq=10, time=np.arange(0, 1, 0.01)):
        self.var["fringe_freq"] = fringe_freq
        if not self.data:
            self.data = experiment.T2Ramsey(soccfg=self.soccfg, soc=self.soc, var=self.var, data_path=self.data_path, title=f"(auto.Ramsey) {int(self.var['r_freq'])}", time=time).run(silent=self.silent).data.T
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

def pi_pulse(var, soccfg=None, soc=None, data_path=None):
    """
    maximize amplitude
    provide value: all variables for pi_pulse
    update value: q_freq, q_length
    return: True for success
    """
    var["q_gain"] = 30000
    var["q_length"] = 0
    def freq_scan(cycle, Δ, nop):
        data = experiment.Rabi(soccfg=soccfg, soc=soc, var=var, data_path=data_path, title=f"(auto.pi_pulse) {int(var['r_freq'])} freq cycle={cycle}", q_freq=np.linspace(var["q_freq"] - Δ, var["q_freq"] + Δ, nop), **{ "2_reps": cycle * 2 }).run().data.T
        F, A = data[0], data[2]
        plt.clf()
        var["q_freq"] = float(helper.symmetryCenter(F, A))
        plt.plot(F, A, marker="o")
        plt.vlines(var["q_freq"], ymin=np.min(A), ymax=np.max(A), colors="red")
        plt.xlabel("Qubit Frequency (MHz)")
        plt.ylabel("Amplitude [lin mag]")
        plt.title(f"Rabi (cycle = {cycle})")
        plt.show()
        print(var["q_freq"], var["q_length"])
    def length_scan(cycle, Δ, nop):
        data = experiment.Rabi(soccfg=soccfg, soc=soc, var=var, data_path=data_path, title=f"(auto.pi_pulse) {int(var['r_freq'])} length cycle={cycle}", q_length=np.linspace(max(var["q_length"] - Δ, 0.01), var["q_length"] + Δ, nop), **{ "2_reps": 2 * cycle }).run().data.T
        L, A = data[0], data[2]
        def m(x, p1, p2, p3):
            return p1 * np.sin(x * p2) ** 2 + p3
        popt, pcov = curve_fit(m, L, A, p0=[np.max(A) - np.min(A), helper.estimateOmega(L, A) / 2, np.min(A)], bounds=([0.1, 6, 0.1], [np.inf, np.inf, np.inf]))
        T = np.pi / popt[1]
        var["q_length"] = float(int(var["q_length"] / T) * T + T / 2)
        plt.clf()
        plt.plot(L, A, marker="o")
        plt.plot(L, m(L, *popt), color="red")
        plt.xlabel("Pi Pulse Length (us)")
        plt.ylabel("Amplitude [lin mag]")
        plt.title(f"Rabi (cycle = {cycle})")
        plt.show()
        print(var["q_freq"], var["q_length"])
    length_scan(0, 0.5, 101)
    freq_scan(10, 10, 71)
    length_scan(1, 0.08, 51)
    freq_scan(10, 5, 51)
    length_scan(10, 0.04, 41)
    return True

def readout(var, soccfg=None, soc=None):
    def negative_fidelity(x):
        var["r_power"], var["r_length"], var["r_offset"] = float(x[0]), float(x[1]), float(x[2])
        data = experiment.IQScatter(soccfg=soccfg, soc=soc, var=var).run(silent=True).data.T
        _, _, _, Fg, Fe, _ = helper.iq_scatter(data[0] + 1j * data[1], data[2] + 1j * data[3])
        plt.close()
        print(x, min(Fg, Fe))
        return -min(Fg, Fe)
    x0 = [var["r_power"], var["r_length"], var["r_offset"]] 
    bounds = [(var["r_power"] - 10, var["r_power"] + 10), (0.1, 5), (0, 2)]
    minimize(negative_fidelity, x0=[var["r_power"], var["r_length"], var["r_offset"]], method="Nelder-Mead", options={ "maxfev": 100 })
    return True

