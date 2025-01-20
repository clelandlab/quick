import numpy as np
from scipy.optimize import minimize, curve_fit, leastsq, least_squares
from scipy import interpolate, stats
import matplotlib.pyplot as plt
import yaml, json
from tqdm.notebook import tqdm
import os, re, ast, configparser
from datetime import datetime
import Pyro4
from qick import QickConfig

_soccfg, _soc = None, None
π = np.pi

yaml.add_representer(np.ndarray, lambda dumper, array: dumper.represent_sequence('tag:yaml.org,2002:seq', array.tolist(), flow_style=True))
yaml.add_multi_representer(np.generic, lambda dumper, data: dumper.represent_data(data.item()))

def connect(ip, port=8888, proxy_name="qick"):
    global _soc, _soccfg
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
    ns = Pyro4.locateNS(host=ip, port=port)
    _soc = Pyro4.Proxy(ns.lookup(proxy_name))
    _soc.get_cfg()
    _soccfg = QickConfig(_soc.get_cfg())
    return _soccfg, _soc

def getSoc(): # Return cached connection
    return _soccfg, _soc

def print_yaml(data): # Print in yaml format
    print(yaml.dump(data))

def load_yaml(path):
    file = open(path, "r")
    content = yaml.safe_load(file)
    file.close()
    return content

def save_yaml(path, data):
    file = open(path, "w")
    content = yaml.dump(data)
    file.write(content)
    return content

def load_data(*paths):
    data_list = []
    for p in paths:
        d = np.genfromtxt(p, delimiter=",")
        data_list.append(d)
    return np.concatenate(data_list)

class Sweep:
    """ Sweeping Tool """
    def __init__(self, config, sweepConfig, progressBar=True):
        self.config = dict(config) # Copy config, avoid changing the original
        self.sweep = [] # [{ 'key': 'key', 'list': [value list], 'index': 0 }]
        self.done = False
        self.progressBar = progressBar
        self.total = 1
        try:
            for k, v in sweepConfig.items():
                l = list(v)
                self.total = self.total * len(l)
                self.sweep.append({ 'key': k, 'list': l, 'index': 0 })
            self.sweep.reverse() # Match Labrad sweeping sequence
        except:
            print("Invalid Sweep Iterator. \n ")
        
    def __iter__(self): # initialze all iterator
        self.done = False
        self.start = False
        if self.progressBar:
            self.progress = tqdm(total=self.total, desc='quick.Sweep')
        for s in self.sweep:
            s['index'] = 0
        return self
        
    def __next__(self):
        if self.progressBar and self.start:
            self.progress.update()
        if self.done:
            if self.progressBar:
                self.progress.close()
            raise StopIteration
        for s in self.sweep: # Always assemble config first!
            self.config[s['key']] = s['list'][s['index']]
        self.done = True
        for s in self.sweep:
            if s['index'] < len(s['list']) - 1:
                s['index'] += 1
                self.done = False
                break
            s['index'] = 0
        self.start = True
        return self.config

class Saver:
    """ Save data in Labrad-compatible format """
    def __init__(self, title, path, indep_params=[], dep_params=[], params={}):
        self.indep_params = []
        self.dep_params = []
        self.params = params
        self.title = title
        self.path = path
        self.file_name = ''
        os.path.exists(path)
        try: # check variables
            for value in indep_params:
                self.indep_params.append([value[0], value[1]])
            for value in dep_params:
                self.dep_params.append([value[0], value[1]])
        except KeyError:
            print("Variables incorrectly formatted. Variable: ('label', 'unit')")
        # Detect max existing file number
        existing_files = [f for f in os.listdir(self.path) if re.search(r"^\d\d\d\d\d \- .*\.csv$", f)]
        if existing_files:
            existing_numbers = [int(f.split(' - ')[0]) for f in existing_files]
            next_number = max(existing_numbers) + 1
        else:
            next_number = 0
        # Creating file name + path based off ^
        self.file_name = self.path + f"/{next_number:05d} - {self.title}"
        self.created_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        self.write_ini()
        self.write_yml()
        # Create empty csv
        np.savetxt(self.file_name + '.csv', [], delimiter=',')

    def write_ini(self):
        # Making FILLED ini
        config = configparser.ConfigParser()
        config["General"] = {
            "created": self.created_time,
            "accessed": self.created_time,
            "modified": self.created_time,
            "title": self.title,
            "independent": len(self.indep_params),
            "dependent": len(self.dep_params),
            "parameters": len(self.params),
            "comments": 0 # Can add comment compatability if people want
        }
        for i in range(0, len(self.indep_params)): # Independents
            config["Independent " + str(i + 1)] = { "label": self.indep_params[i][0], "units": self.indep_params[i][1] }
        for i in range(0, len(self.dep_params)): # Dependents
            config["Dependent " + str(i + 1)] = { "label": self.dep_params[i][0], "units": self.dep_params[i][1] }
        config["Parameter 1"] = { "label": "deprecated", "data": "meta_information_in_yml_file" }
        config["Comments"] = {} # No comments bc nobody uses them
        # Making file
        with open(self.file_name + ".ini", "w") as configfile:
            config.write(configfile)

    def write_yml(self):
        meta = {
            "created": self.created_time,
            "completed": datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
            "title": self.title,
            "independent": self.indep_params,
            "dependent": self.dep_params,
            "parameters": self.params,
            "comments": []
        }
        save_yaml(self.file_name + ".yml", meta)

    def write_data(self, data):
        # Todo: check data dimension
        with open(self.file_name + '.csv', 'a+') as f:
            np.savetxt(f, data, fmt="%.9e", delimiter=',')
        # self.write_yml() # do not call it every time for efficiency

dbm_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "./constants/dbm.npy")
dbm_data = np.load(dbm_path)

def dbm2gain(dbm, freq, nqz, balun):
    f = interpolate.interp1d(np.linspace(1e8, 8e9, 80), dbm_data[balun*2 + nqz - 1])
    dbm3e5 = f(freq*1e6)
    return 10**((dbm - dbm3e5)/20)

def evalStr(s, var):
    return eval(f"f'''{s}'''", None, var)

def symmetryCenter(x, y, it=3):
    L = len(y)
    R = int(L / 2)
    Δ = int(R / 2)
    freq = np.fft.rfftfreq(L*2)
    y = (y - np.mean(y)) / np.std(y)
    fft = np.fft.rfft(np.append(y, np.zeros(L)))
    res, d = 0, 0
    for _ in range(it):
        fft *= np.exp(2j*π*d*freq)
        co = np.fft.irfft(fft**2)[R-1:L+R-1]
        i = np.argmax(co[R-Δ:R+Δ]) + R-Δ
        p = np.polynomial.polynomial.polyfit(np.arange(i-2, i+3), co[i-2:i+3], 2)
        d = -p[1]/4/p[2] - R/2
        res += d
    i = res + R - 0.5
    _i = int(i)
    return x[_i] + (x[_i+1] - x[_i]) * (i - _i)

def estimateOmega(x, y):
    y = (y - np.mean(y)) / np.std(y)
    rfft = np.fft.rfft(y)
    freq = np.fft.rfftfreq(len(y))
    return len(y) * freq[np.argmax(np.abs(rfft))] * 2 * π / (x[len(x) - 1] - x[0])

# IQ scatter and centering
def iq2prob(Ss, c0, c1):
    """convert IQ raw data to probability, according to the |0> center and |1> center."""
    center0, center1 = [np.real(c0), np.imag(c0)], [np.real(c1), np.imag(c1)]
    Is_shift = np.real(Ss) - (center0[0] + center1[0]) / 2.0
    Qs_shift = np.imag(Ss) - (center0[1] + center1[1]) / 2.0
    angle = np.angle(-1j * center1[1] - center1[0] + 1j * center0[1] + center0[0])
    rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    I_rot = np.dot(rot_mat, np.vstack([Is_shift, Qs_shift]))[0]
    return float(I_rot[I_rot < 0].size) / I_rot.size

# rotation angle and real threshold after rotation.
def iq_rotation(c0, c1):
    return float(-np.angle(c1 - c0, deg=True)), float(np.real((c1 + c0) / 2 * np.exp(-1j *  np.angle(c1 - c0))))

# return c0, c1, visibility, Fg, Fe, fig
def iq_scatter(S0s, S1s, c0=None, c1=None):
    if c0 is None:
        c0 = np.median(S0s.real) + 1j * np.median(S0s.imag)
    if c1 is None:
        c1 = np.median(S1s.real) + 1j * np.median(S1s.imag)
    def negative_visibility(_p):
        p = _p[0] + 1j * _p[1]
        angle = np.angle(c0 - p)
        S0 = (S0s - (p + c0) / 2.0) * np.exp(-1j * angle)
        S1 = (S1s - (p + c0) / 2.0) * np.exp(-1j * angle)
        prob0 = float(S0.real[S0.real < 0].size) / S0.size
        prob1 = float(S1.real[S1.real < 0].size) / S1.size
        return prob0 - prob1
    res = minimize(negative_visibility, [c1.real, c1.imag], method='Nelder-Mead') # optimize c1
    visibility = -1 * negative_visibility(res.x)
    c1 = res.x[0] + 1j * res.x[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # plot the histogram
    o = (c0 + c1) / 2.0 # origin
    g = c0 - o # project on g
    S0p = np.real((S0s - o) * np.conj(g)) / np.abs(g)
    S1p = np.real((S1s - o) * np.conj(g)) / np.abs(g)
    S0p_right, S1p_right = S0p[S0p > 0], S1p[S1p < 0]
    S0s_right, S0s_wrong = S0s[S0p > 0], S0s[S0p <= 0]
    S1s_right, S1s_wrong = S1s[S1p < 0], S1s[S1p >= 0]
    Fg, Fe = len(S0p_right) / len(S0p), len(S1p_right) / len(S1p)
    xp = np.linspace(np.min(S1p), np.max(S0p), 2001) # axis of projection
    g_hist, _ = np.histogram(S0p, bins=100, density=True, range=[xp[0], xp[-1]])
    e_hist, _ = np.histogram(S1p, bins=100, density=True, range=[xp[0], xp[-1]])
    g_cdf = np.cumsum(g_hist) * (xp[-1] - xp[0]) / 100
    e_cdf = np.cumsum(e_hist) * (xp[-1] - xp[0]) / 100
    axes[1].hist(S0p, color='blue', alpha=0.5, bins=100, density=True, range=[xp[0], xp[-1]], label=r'$|g\rangle$')
    axes[1].hist(S1p, color='red', alpha=0.5, bins=100, density=True, range=[xp[0], xp[-1]], label=r'$|e\rangle$')
    g_mu, g_std = stats.norm.fit(S0p_right)
    axes[1].plot(xp, stats.norm.pdf(xp, g_mu, g_std), color='blue', linestyle='dotted')
    e_mu, e_std = stats.norm.fit(S1p_right)
    axes[1].plot(xp, stats.norm.pdf(xp, e_mu, e_std), color='red', linestyle='dotted')
    axr = axes[1].twinx()
    axr.set_ylim([0, 1])
    axr.set_ylabel("Cumulative Probability")
    axr.plot(np.linspace(xp[0], xp[-1], 100), g_cdf, color='blue', linewidth=2)
    axr.plot(np.linspace(xp[0], xp[-1], 100), e_cdf, color='red', linewidth=2)
    axr.plot(np.linspace(xp[0], xp[-1], 100), e_cdf - g_cdf, color='green', linewidth=2)
    axes[1].set_xlabel('Projected IQ Amplitude (a.u.)')
    axes[1].set_ylabel('Probability')
    axes[1].legend(shadow=False, loc=1, frameon=True)
    axes[1].grid()
    axes[1].text(0.05, 0.95, 'Visibility=%.3f' % visibility, ha='left', transform=axes[1].transAxes)
    axes[1].text(0.05, 0.9, '$F_g$=%.3f' % Fg, ha='left', transform=axes[1].transAxes)
    axes[1].text(0.05, 0.85, '$F_e$=%.3f' % Fe, ha='left', transform=axes[1].transAxes)
    # scatter plot
    axes[0].scatter(S0s_right.real, S0s_right.imag, s=10, marker='o', zorder=0, alpha=0.5, c='blue', linewidths=0.0)
    axes[0].scatter(S1s_right.real, S1s_right.imag, s=10, marker='o', zorder=0, alpha=0.5, c='red', linewidths=0.0)
    axes[0].scatter(S0s_wrong.real, S0s_wrong.imag, s=10, marker='o', zorder=0, alpha=0.5, c='blue', linewidths=0.0)
    axes[0].scatter(S1s_wrong.real, S1s_wrong.imag, s=10, marker='o', zorder=0, alpha=0.5, c='red', linewidths=0.0)
    axes[0].grid()
    axes[0].set_xlabel('I (a.u.)')
    axes[0].set_ylabel('Q (a.u.)')
    axes[0].set_xlim(min(np.min(S0s.real), np.min(S1s.real)) - 10, max(np.max(S0s.real), np.max(S1s.real)) + 10)
    axes[0].set_ylim(min(np.min(S0s.imag), np.min(S1s.imag)) - 10, max(np.max(S0s.imag), np.max(S1s.imag)) + 10)
    # plot the cut
    _x = np.linspace(min(np.min(S0s.real), np.min(S1s.real)) - 10, max(np.max(S0s.real), np.max(S1s.real)) + 10, 101)
    axes[0].plot( _x, -(c0.real - c1.real) / (c0.imag - c1.imag) * (_x - (c0.real + c1.real) / 2) + (c0.imag + c1.imag) / 2, 'k--')
    # add the circle to the IQ scatter plot
    theta = np.linspace(-π, π, 1001)
    I_g_fit_list = np.median(S0s_right.real) + np.cos(theta) * 3 * g_std
    Q_g_fit_list = np.median(S0s_right.imag) + np.sin(theta) * 3 * g_std
    I_e_fit_list = np.median(S1s_right.real) + np.cos(theta) * 3 * e_std
    Q_e_fit_list = np.median(S1s_right.imag) + np.sin(theta) * 3 * e_std
    axes[0].plot(I_g_fit_list, Q_g_fit_list, color='darkblue', linestyle='--', label=r'$|g\rangle$, $3\sigma$')
    axes[0].plot(I_e_fit_list, Q_e_fit_list, color='darkred', linestyle='--', label=r'$|e\rangle$, $3\sigma$')
    axes[0].legend(shadow=False, loc=1, frameon=True)
    return c0, c1, visibility, Fg, Fe, fig

# T2 fit and plot
def fitT2(T, S, omega=2*π):
    def m(x, *p):
        return p[0] * np.exp(-x / p[1]) * np.cos(p[2] * x) + p[3]
    def jac(x, *p):
        d0 = np.exp(-x / p[1]) * np.cos(p[2] * x)
        d1 = p[0] * x / p[1] ** 2 * np.exp(-x / p[1]) * np.cos(p[2] * x)
        d2 = - x * p[0] * np.exp(-x / p[1]) * np.sin(p[2] * x)
        d3 = np.ones(len(x))
        return np.transpose([d0, d1, d2, d3])
    def me(x, *p):
        return p[0] * np.exp(-x / p[1]) + p[3]
    p0 = [0.4, 20.0, omega, 0.5]
    popt, pcov = curve_fit(m, T, S, p0=p0, jac=jac)
    perr = np.sqrt(np.diag(pcov))  # Standard deviation of parameters
    residuals = S - m(T, *popt)
    ss_res = np.sum(residuals**2) / np.var(S)
    dof = len(T) - len(popt)
    rchi2 = ss_res / dof
    fig, ax = plt.subplots()
    ax.scatter(T, S, color='black', s=10, label='Original Data')
    ax.plot(T, m(T, *popt), color='red', label='Fit')
    ax.plot(T, me(T, *popt), color='blue', label='Fit Exponential')
    annotation_text = (
        r"$S = S_0e^{-\tau/T_2}\cos(\omega\tau) + C$" + "\n" +
        "Fitting Result:\n" +
        r"$S_0 = ({:.2f} \pm {:.2f})$, $C = ({:.2f} \pm {:.2f})$".format(popt[0], perr[0], popt[3], perr[3]) + "\n" +
        r"$T_2 = ({:.2f} \pm {:.2f})$ us".format(popt[1], perr[1]) + "\n" +
        r"$\omega = ({:.2f} \pm {:.2f})$ rad/us".format(popt[2], perr[2]) + "\n" +
        r"$n = {}$ (DOF), $\chi^2/n = {:.2e}$".format(dof, rchi2)
    )
    ax.set_xlabel(r"Pulse Delay $\tau$ (us)")
    ax.set_ylabel("Population")
    ax.set_title("T2 Measurement")
    ax.grid()
    ax.legend(loc="lower right")
    mid_index = len(T) // 2
    ax.annotate(annotation_text, (T[mid_index], me(T[mid_index], *popt)), fontsize=8, xycoords='data', textcoords='offset points', xytext=(10, 10))
    return popt, perr, rchi2, fig

def fitResonator(F, S, fit="circle", p0=[None, None, None, None, None, None, None]):
    def db(a):
        return 20. * np.log10(np.abs(a))
    def background_noise(p, f):
        Qi, Qc, fr, phi, electronic_delay, background, phase_shift = p
        return 10**(background / 20) * np.exp(1j * (-2 * π * f * electronic_delay + phase_shift))
    def S21_th(f, *p):
        Qi, Qc, fr, phi, electronic_delay, background, phase_shift = p
        return background_noise(p, f) / (1 + Qi / Qc * np.exp(1j * phi) / (1 + 2j * Qi * (f - fr) / f))
    s = F.argsort()
    F, S = F[s], S[s]
    S_inv, S_db = 1 / S, db(S)
    _p0 = [100000, 10000, F[np.argmin(S_db)], 0.0, np.polyfit(-2 * π * F[0:100], np.unwrap(np.angle(S[0:100])), deg=1)[0], (S_db[0] + S_db[-1]) / 2, 0.0]
    p0 = list(p0)
    for i in range(len(p0)):
        if p0[i] is None:
            p0[i] = _p0[i]
    bounds = ([0, 0, 0, -π, p0[4]*0.95, p0[5] - 10, 0], [np.inf, np.inf, np.inf, π, p0[4]*1.05, p0[5] + 10, 2 * π])
    def p_b2i(p):
        _p = []
        for i in range(len(p)):
            if bounds[1][i] == np.inf:
                _p.append(np.sqrt((p[i] - bounds[0][i] + 1) ** 2 - 1))
            else:
                _p.append(np.arcsin(2 * (p[i] - bounds[0][i]) / (bounds[1][i] - bounds[0][i]) - 1))
        return _p
    def p_i2b(_p):
        p = []
        for i in range(len(_p)):
            if bounds[1][i] == np.inf:
                p.append(bounds[0][i] - 1 + np.sqrt(_p[i] ** 2 + 1))
            else:
                p.append(bounds[0][i] + (np.sin(_p[i]) + 1) * (bounds[1][i] - bounds[0][i]) / 2)
        return p
    def S21_circle_fit_err(_p):
        y = 1 / S21_th(F, *p_i2b(_p))
        return np.concatenate((np.real(S_inv - y), np.imag(S_inv - y)))
    def S21_amp_fit_err(_p):
        return db(S21_th(F, *p_i2b(_p))) - S_db
    def S21_arg_fit_err(p):
        return np.angle(S / S21_th(F, *p))
    if fit == "circle":
        result = leastsq(S21_circle_fit_err, p_b2i(p0), maxfev=50000, xtol=1.e-7, ftol=1.e-7, col_deriv=False, gtol=1.e-7)
        p = p_i2b(result[0])
    elif fit == "amp":
        bounds[0][4], bounds[1][4] = p0[4] - 0.01, p0[4] + 0.01
        bounds[0][6], bounds[1][6] = p0[6] - 0.01, p0[6] + 0.01
        result = leastsq(S21_amp_fit_err, p_b2i(p0), maxfev=50000, xtol=1.e-7, ftol=1.e-7, col_deriv=False, gtol=1.e-7)
        p = p_i2b(result[0])
    else:
        bounds[0][5], bounds[1][5] = p0[5] - 0.01, p0[5] + 0.01
        result = least_squares(S21_arg_fit_err, p0, bounds=bounds)
        p = result.x
    S21_fit = S21_th(F, *p)
    back_noise = background_noise(p, F)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True, sharex=False)
    axes[0].scatter((1 / S * back_noise).real, (1 / S * back_noise).imag, label="raw", alpha=0.5, linewidths=0.0)
    axes[0].plot((1 / S21_fit * back_noise).real, (1 / S21_fit * back_noise).imag, "r-", label="fit")
    axes[0].set_xlabel(r"Re $\widetilde{S}_{21}^{-1}$")
    axes[0].set_ylabel(r"Im $\widetilde{S}_{21}^{-1}$")
    axes[1].scatter(F, S_db, alpha=0.5, linewidths=0.0)
    axes[1].plot(F, db(S21_fit), "r-")
    axes[1].set_xlim(F[0], F[-1])
    axes[1].set_xlabel(r"$f$ (MHz)")
    axes[1].set_ylabel(r"$|S_{21}|$(dB)")
    axes[1].text(0.7, 0.29, r"Qi=%d" % p[0], ha="left", va="bottom", transform=axes[1].transAxes)
    axes[1].text(0.7, 0.21, r"Qc=%d" % p[1], ha="left", va="bottom", transform=axes[1].transAxes)
    axes[1].text(0.7, 0.13, r"$f$=%.4f" % p[2], ha="left", va="bottom", transform=axes[1].transAxes)
    axes[1].text(0.7, 0.05, r"bg=%.1fdB" % p[5], ha="left", va="bottom", transform=axes[1].transAxes)
    axes[2].scatter(F, np.unwrap((np.angle(S) + 2*π) % (2*π)), alpha=0.5, linewidths=0.0)
    axes[2].plot(F, np.unwrap((np.angle(S21_fit) + 2*π) % (2*π)), 'r-')
    axes[2].set_xlim(F[0], F[-1])
    axes[2].set_xlabel(r"$f$ (GHz)")
    axes[2].set_ylabel(r"$arg~S_{21}$")
    return p, fig
