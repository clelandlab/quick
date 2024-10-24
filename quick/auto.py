import quick.experiment as experiment
import quick.helper as helper
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import convolve1d, median_filter
from scipy.signal import find_peaks
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

class BaseAuto:
    var = {}
    data = [[]]
    def __init__(self, var):
        self.var = dict(var)
    def measure(self):
        pass
    def load_data(self, *paths):
        self.data = helper.load_data(*paths).T
    def calibrate(self):
        return False, None
    def check(self):
        return True

class Resonator(BaseAuto):
    def measure(self, silent=False, data_path=None, soccfg=None, soc=None):
        self.var["r_relax"] = 1
        self.data = experiment.ResonatorSpectroscopy(data_path=data_path, title=f'(auto.Resonator) {int(self.var["r_freq"])}', r_power=np.arange(-60, -15, 1), r_freq=np.linspace(self.var["r_freq"] - 2, self.var["r_freq"] + 2, 100), soccfg=soccfg, soc=soc, var=self.var).run(silent=silent).data.T
        return self.data
    def calibrate(self, silent=False):
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

def q_freq(var, span=[3000, 5000], soccfg=None, soc=None, data_path=None):
    """
    find qubit frequency by peak finding in QubitSpectroscopy
    provide value: r_freq, r_offset, r_power, r_length, q_gain
    provide center: q_freq
    update value: q_freq
    return: True for success
    """
    def scan(r, nop, w=5, mean=False):
        data = experiment.QubitSpectroscopy(data_path=data_path, title=f'(auto.q_freq) {int(var["r_freq"])}', q_freq=np.linspace(var["q_freq"] - r, var["q_freq"] + r, nop), soccfg=soccfg, soc=soc, var=var).run().data.T
        F, A = data[0], data[1]
        C = np.convolve(A, np.ones(w), 'same') / w if mean else median_filter(A, size=w)
        plt.clf()
        plt.scatter(F, A, s=2, label="Data")
        plt.plot(F[w:-w], C[w:-w], color="red", label=f"{'Mean' if mean else 'Median'} w={w}")
        plt.xlabel("Qubit Frequency (MHz)")
        plt.ylabel("Amplitude [lin mag]")
        plt.title(f"Qubit Spectroscopy (gain = {var['q_gain']})")
        plt.legend()
        plt.show()
        return F, A, C
    var["q_length"] = 2
    var["q_freq"] = (span[1] + span[0]) / 2
    r = (span[1] - span[0]) / 2
    F, A, C = scan(r, int(r), 3)
    var["q_freq"] = float(F[np.argmax(C)])
    var["q_gain"] = 3000
    F, A, C = scan(20, 100, 5, mean=True)
    var["q_freq"] = helper.symmetryCenter(F, C)

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

def r_freq(var, soccfg=None, soc=None, data_path=None):
    v = dict(var)
    data = experiment.DispersiveSpectroscopy(soccfg=soccfg, soc=soc, var=v, data_path=data_path, title=f"(auto.r_freq) {int(v['r_freq'])}", r_freq=np.arange(v["r_freq"] - 1, v["r_freq"] + 1, 0.02)).run().data.T
    F, A0, P0, A1, P1 = data[0], data[1], data[2], data[5], data[6]
    dP = np.convolve(np.unwrap(P0 - P1), [0.333, 0.333, 0.333], "same")
    v["r_freq"] = float(F[np.argmax(dP)])
    plt.clf()
    plt.plot(F, A0, label="Amplitude 0", marker="o")
    plt.plot(F, A1, label="Amplitude 1", marker="o")
    plt.vlines(v["r_freq"], ymin=np.min(A0), ymax=np.max(A0), colors="red", label="Max Phase Diff.")
    plt.legend()
    plt.xlabel("Readout Frequency (MHz)")
    plt.ylabel("Amplitude [log mag] (dB)")
    plt.title("DispersiveSpectroscopy")
    plt.show()
    return v

def ramsey(var, soccfg=None, soc=None, data_path=None):
    v = dict(var)
    v["r_phase"] = 0
    data = experiment.IQScatter(soccfg=soccfg, soc=soc, var=v).run().data.T
    c0, c1, _, _, _, _ = helper.iq_scatter(data[0] + 1j * data[1], data[2] + 1j * data[3])
    v["r_phase"], v["r_threshold"] = helper.iq_rotation(c0, c1)
    data = experiment.T2Ramsey(soccfg=soccfg, soc=soc, var=v, data_path=data_path, title=f"(auto.ramsey) {int(v['r_freq'])}", time=np.arange(0, 1, 0.01)).run().data.T
    L, A = data[0], data[1]
    def m(x, p1, p2, p3):
        return p1 * np.cos(p2 * x) + p3
    popt, pcov = curve_fit(m, L, A, p0=[(np.max(A) - np.min(A)) / 2, helper.estimateOmega(L, A), 0.5])
    plt.clf()
    plt.scatter(L, A, s=10, color="black")
    plt.plot(np.arange(0, 1, 0.001), m(np.arange(0, 1, 0.001), *popt), color="red")
    plt.title("Ramsey (fringe_freq = 10)")
    plt.show()
    v["q_freq"] = float(v["q_freq"] + popt[1] / 2 / np.pi - 10)
    print(v["q_freq"], popt[1] / 2 / np.pi - 10)
    return v

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

