import quick.experiment as experiment
import quick.helper as helper
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import convolve1d, median_filter
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

def resonator(var, r=2, soccfg=None, soc=None, data_path=None):
    """
    find resonator frequency and power
    provide center: r_freq
    update value: r_freq, r_power
    return: True for success
    """
    data = experiment.ResonatorSpectroscopy(data_path=data_path, title=f'(auto.resonator) {int(var["r_freq"])}', r_freqs=np.linspace(var["r_freq"] - r, var["r_freq"] + r, 100), r_powers=np.arange(-55, -14, 1), soccfg=soccfg, soc=soc, var=var).run().data.T
    P, F, A = data[0], data[1], data[2]
    F = F[0:100]
    P = P[0::100]
    A = A.reshape((-1, 100))
    A = convolve1d(A, [0.333, 0.333, 0.333], axis=1)
    M = np.min(A, axis=1)
    stable = M[-1]
    if abs(np.max(M) - stable) < 4:
        return False
    index = -1
    for i in range(np.argmax(M), 0, -1):
        if abs(M[i] - stable) < 1:
            index = i
            break
    if index < 0:
        return False
    var["r_power"] = float(P[index])
    var["r_freq"] = float(F[np.argmin(A[index])])
    return True

def q_freq(var, span=[3000, 5000], soccfg=None, soc=None, data_path=None):
    """
    find qubit frequency by peak finding in QubitSpectroscopy
    provide value: r_freq, r_offset, r_power, r_length, q_gain
    provide center: q_freq
    update value: q_freq
    return: True for success
    """
    def scan(r, nop, w=5, mean=False):
        data = experiment.QubitSpectroscopy(data_path=data_path, title=f'(auto.q_freq) {int(var["r_freq"])}', q_freqs=np.linspace(var["q_freq"] - r, var["q_freq"] + r, nop), soccfg=soccfg, soc=soc, var=var).run().data.T
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
        data = experiment.Rabi(soccfg=soccfg, soc=soc, var=var, data_path=data_path, title=f"(auto.pi_pulse) {int(var['r_freq'])} freq cycle={cycle}", q_freqs=np.linspace(var["q_freq"] - Δ, var["q_freq"] + Δ, nop), **{ "2_reps": cycle * 2 }).run().data.T
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
        data = experiment.Rabi(soccfg=soccfg, soc=soc, var=var, data_path=data_path, title=f"(auto.pi_pulse) {int(var['r_freq'])} length cycle={cycle}", q_lengths=np.linspace(max(var["q_length"] - Δ, 0.01), var["q_length"] + Δ, nop), **{ "2_reps": 2 * cycle }).run().data.T
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
    data = experiment.DispersiveSpectroscopy(soccfg=soccfg, soc=soc, var=var, data_path=data_path, title=f"(auto.r_freq) {int(var['r_freq'])}", r_freqs=np.arange(var["r_freq"] - 1, var["r_freq"] + 1, 0.02)).run().data.T
    F, A0, P0, A1, P1 = data[0], data[1], data[2], data[5], data[6]
    dP = np.convolve(np.unwrap(P0 - P1), [0.333, 0.333, 0.333], "same")
    var["r_freq"] = float(F[np.argmax(dP)])
    plt.clf()
    plt.plot(F, A0, label="Amplitude 0", marker="o")
    plt.plot(F, A1, label="Amplitude 1", marker="o")
    plt.vlines(var["r_freq"], ymin=np.min(A0), ymax=np.max(A0), colors="red", label="Max Phase Diff.")
    plt.legend()
    plt.xlabel("Readout Frequency (MHz)")
    plt.ylabel("Amplitude [log mag] (dB)")
    plt.title("DispersiveSpectroscopy")
    plt.show()
    return True

def ramsey(var, soccfg=None, soc=None, data_path=None):
    var["r_phase"] = 0
    data = experiment.IQScatter(soccfg=soccfg, soc=soc, var=var).run().data.T
    c0, c1, _, _, _, _ = helper.iq_scatter(data[0] + 1j * data[1], data[2] + 1j * data[3])
    var["r_phase"], var["r_threshold"] = helper.iq_rotation(c0, c1)
    var["fringe_freq"] = 10
    data = experiment.T2Ramsey(soccfg=soccfg, soc=soc, var=var, data_path=data_path, title=f"(auto.ramsey) {int(var['r_freq'])}", times=np.arange(0, 1, 0.01)).run().data.T
    L, A = data[0], data[1]
    def m(x, p1, p2, p3):
        return p1 * np.cos(p2 * x) + p3
    popt, pcov = curve_fit(m, L, A, p0=[(np.max(A) - np.min(A)) / 2, helper.estimateOmega(L, A), 0.5])
    plt.clf()
    plt.scatter(L, A, s=10, color="black")
    plt.plot(np.arange(0, 1, 0.001), m(np.arange(0, 1, 0.001), *popt), color="red")
    plt.title("Ramsey (fringe_freq = 10)")
    plt.show()
    var["q_freq"] = float(var["q_freq"] + popt[1] / 2 / np.pi - 10)
    print(var["q_freq"], popt[1] / 2 / np.pi - 10)

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

