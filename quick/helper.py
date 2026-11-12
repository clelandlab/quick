"""QuICK helpers and the local configuration used by its board client.

The configuration conversions follow QICK 0.2.418.
Derived portions are Copyright (c) Open Quantum Hardware, under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from scipy.optimize import minimize, curve_fit, leastsq
from scipy import interpolate, stats
import yaml, json, os, re, copy
from tqdm.notebook import tqdm
from func_timeout import func_timeout, FunctionTimedOut
from datetime import datetime
import logging


class _SocConfig:
    """Local RFSoC configuration and unit conversions, with no board dependency.

    The numerical conventions match the 0.2.418 firmware client: MHz, degrees,
    microseconds, nearest-even register rounding, and absolute v2 frequencies.
    Only configuration dictionaries and numeric arrays cross the connection.
    """

    def __init__(self, cfg=None):
        if isinstance(cfg, (str, os.PathLike)):
            with open(cfg) as f:
                cfg = json.load(f)
        if hasattr(cfg, "get_cfg"):
            cfg = cfg.get_cfg()
        self._cfg = {} if cfg is None else cfg

    def __getitem__(self, key):
        return self._cfg[key]

    def __setitem__(self, key, value):
        self._cfg[key] = value

    def __contains__(self, key):
        return key in self._cfg

    def __iter__(self):
        return iter(self._cfg)

    def __len__(self):
        return len(self._cfg)

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def keys(self):
        return self._cfg.keys()

    def items(self):
        return self._cfg.items()

    def get_cfg(self):
        return self._cfg

    def dump_cfg(self, gen_file_path=None):
        result = json.dumps(self._cfg, indent=4)
        if gen_file_path is not None:
            with open(gen_file_path, "w") as f:
                f.write(result)
        return result

    def description(self):
        cfg = self._cfg
        tproc = next(iter(cfg.get("tprocs", [])), {})
        lines = ["QuICK running on %s, software version %s" % (
            cfg.get("board", "RFSoC"), cfg.get("sw_version", "unknown")),
            "\nFirmware configuration (built %s):" % cfg.get("fw_timestamp", "unknown")]
        if "refclk_freq" in cfg:
            lines.append("\n\tGlobal clocks (MHz): tProc dispatcher timing %.3f, RF reference %.3f" % (
                tproc.get("f_time", 0), cfg["refclk_freq"]))

        lines.append("\n\t%d signal generator channels:" % len(cfg.get("gens", [])))
        for i, gen in enumerate(cfg.get("gens", [])):
            lines.append("\t%d:\t%s" % (i, gen.get("type", "channel")))
            if "fs" in gen and "f_fabric" in gen:
                lines[-1] += " - fs=%.3f Msps, fabric=%.3f MHz" % (gen["fs"], gen["f_fabric"])
            if all(k in gen for k in ("maxlen", "samps_per_clk", "f_fabric")):
                duration = gen["maxlen"] / (gen["samps_per_clk"] * gen["f_fabric"])
                lines.append("\t\tenvelope memory: %d %s samples (%.3f us)" % (
                    gen["maxlen"], "complex" if gen.get("complex_env", True) else "real", duration))
            if gen.get("has_dds", True) and "b_dds" in gen and "f_dds" in gen:
                lines.append("\t\t%d-bit DDS, range=%.3f MHz" % (gen["b_dds"], gen["f_dds"]))
            if "dac" in gen:
                lines.append("\t\t" + self._describe_converter("DAC", gen["dac"]))

        lines.append("\n\t%d readout channels:" % len(cfg.get("readouts", [])))
        for i, ro in enumerate(cfg.get("readouts", [])):
            control = "tProc output %d" % ro["tproc_ctrl"] if "tproc_ctrl" in ro else "PYNQ"
            lines.append("\t%d:\t%s - configured by %s" % (i, ro.get("ro_type", "channel"), control))
            if all(k in ro for k in ("fs", "f_output", "b_dds", "f_dds")):
                lines.append("\t\tfs=%.3f Msps, decimated=%.3f MHz, %d-bit DDS, range=%.3f MHz" % (
                    ro["fs"], ro["f_output"], ro["b_dds"], ro["f_dds"]))
            if "avgbuf_type" in ro:
                lines.append("\t\t%s v%s (%s edge counter, %s weights)" % (
                    ro["avgbuf_type"], ro.get("avgbuf_version", "unknown"),
                    "has" if ro.get("has_edge_counter") else "no", "has" if ro.get("has_weights") else "no"))
            if all(k in ro for k in ("avg_maxlen", "buf_maxlen", "f_output")):
                lines.append("\t\tmemory %d accumulated, %d decimated (%.3f us)" % (
                    ro["avg_maxlen"], ro["buf_maxlen"], ro["buf_maxlen"] / ro["f_output"]))
                if ro.get("has_weights") and "wgt_maxlen" in ro:
                    lines[-1] += ", %d weights (%.3f us)" % (ro["wgt_maxlen"], ro["wgt_maxlen"] / ro["f_output"])
            if all(k in ro for k in ("trigger_type", "trigger_port", "trigger_bit", "tproc_ch")):
                lines.append("\t\ttriggered by %s %d, pin %d, feedback to tProc input %d" % (
                    ro["trigger_type"], ro["trigger_port"], ro["trigger_bit"], ro["tproc_ch"]))
            if "adc" in ro:
                lines.append("\t\t" + self._describe_converter("ADC", ro["adc"]))

        if tproc:
            pins = tproc.get("output_pins", [])
            lines.append("\n\t%d digital output pins:" % len(pins))
            for i, (kind, port, pin, name) in enumerate(pins):
                lines.append("\t%d:\t%s (%s %d, pin %d)" % (i, name, kind, port, pin))
            kind = tproc.get("type", "unknown")
            version = {"qick_processor": "v2", "axis_tproc64x32_x8": "v1"}.get(kind, "unknown")
            lines.append('\n\ttProc: %s ("%s") rev %s' % (kind, version, tproc.get("revision", "unknown")))
            if "f_core" in tproc:
                lines[-1] += ", core execution clock %.3f MHz" % tproc["f_core"]
            memories = ["%s %d" % (label, tproc[key]) for key, label in (
                ("pmem_size", "program"), ("dmem_size", "data"), ("wmem_size", "waveform")) if key in tproc]
            if memories:
                lines.append("\t\tmemories (words): " + ", ".join(memories))
            for key, label in (("start_pin", "start"), ("stop_pin", "stop")):
                if key in tproc:
                    lines.append("\t\texternal %s pin: %s" % (label, tproc[key]))
        lines.extend(cfg.get("extra_description", []))
        return "\n".join(lines)

    def _describe_converter(self, kind, name):
        if len(str(name)) != 2 or not str(name).isdigit():
            return "%s %s" % (kind, name)
        tile, block = map(int, str(name))
        label = "%s tile %d, blk %d" % (kind, tile, block)
        if self.get("board") == "ZCU216":
            dac = kind == "DAC"
            connector = (1 if dac else 5) + block % 2 + 2 * (tile // 2)
            label += " is %d_%d on JHC%d" % (block, tile + (228 if dac else 224), connector)
            if dac or tile in (1, 2):
                label += ", or QICK box %s port %d" % (kind, (tile if dac else tile - 1) * 4 + block)
        return label

    def __str__(self):
        return self.description()

    @staticmethod
    def _to_int(value, scale, quantize=1, parname=None):
        if hasattr(value, "to_int"):
            return value.to_int(scale, quantize=quantize, parname=parname, trunc=False)
        return int(quantize * np.round(value * scale / quantize))

    def calc_fstep_int(self, dict1, other_dicts):
        channels = [dict1] + list(other_dicts)
        common_div = np.lcm.reduce([d["fdds_div"] for d in channels])
        max_bits = max(d["b_dds"] for d in channels)
        multipliers = [d["fs_mult"] * (common_div // d["fdds_div"]) *
                       2 ** (max_bits - d["b_dds"]) for d in channels]
        return np.lcm.reduce(multipliers) // multipliers[0]

    def ch_fstep(self, channel):
        return channel["fs_mult"] * (self["refclk_freq"] / channel["fdds_div"]) / 2 ** channel["b_dds"]

    def calc_fstep(self, dicts):
        step = self.ch_fstep(dicts[0])
        if len(dicts) > 1:
            step *= self.calc_fstep_int(dicts[0], dicts[1:])
        return step

    def roundfreq(self, f, dicts):
        step = self.calc_fstep(dicts)
        rounded = np.round(f / step) * step
        return float(rounded) if np.isscalar(rounded) else rounded

    def freq2int(self, f, thisch, otherch=None):
        step = 1 if otherch is None else self.calc_fstep_int(thisch, [otherch])
        return self._to_int(f, 1 / self.ch_fstep(thisch), quantize=step, parname="freq")

    def int2freq(self, r, thisch):
        return r / (2 ** thisch["b_dds"] / thisch["f_dds"])

    def freq2reg(self, f, gen_ch=0, ro_ch=None):
        gen = self["gens"][gen_ch]
        ro = None if ro_ch is None else self["readouts"][ro_ch]
        if gen["interpolation"] != 1 and (f > gen["f_dds"] / 2 or f < -gen["f_dds"] / 2):
            raise RuntimeError("requested frequency %f is outside of [-range/2, range/2]" % f)
        return self.freq2int(f, gen, ro) % 2 ** gen["b_dds"]

    def freq2reg_adc(self, f, ro_ch=0, gen_ch=None):
        ro = self["readouts"][ro_ch]
        gen = None if gen_ch is None else self["gens"][gen_ch]
        return self.freq2int(f, ro, gen) % 2 ** ro["b_dds"]

    def reg2freq(self, r, gen_ch=0):
        return self.int2freq(r, self["gens"][gen_ch])

    def reg2freq_adc(self, r, ro_ch=0):
        return self.int2freq(r, self["readouts"][ro_ch])

    def adcfreq(self, f, gen_ch=0, ro_ch=0):
        return self.roundfreq(f, [self["gens"][gen_ch], self["readouts"][ro_ch]])

    def _get_ch_cfg(self, gen_ch=None, ro_ch=None):
        if gen_ch is not None and ro_ch is not None:
            raise RuntimeError("can't specify both gen_ch and ro_ch!")
        if gen_ch is not None:
            return self["gens"][gen_ch]
        if ro_ch is not None:
            return self["readouts"][ro_ch]
        return None

    def _get_mixer_cfg(self, gencfg):
        return {"fs_mult": gencfg["fs_mult"], "fdds_div": gencfg["fs_div"], "b_dds": 48}

    def deg2int(self, deg, thisch):
        return self._to_int(deg, 2 ** thisch["b_phase"] / 360, parname="phase") % 2 ** thisch["b_phase"]

    def int2deg(self, r, thisch):
        return r / (2 ** thisch["b_phase"] / 360)

    def deg2reg(self, deg, gen_ch=0, ro_ch=None):
        ch = self._get_ch_cfg(gen_ch, ro_ch)
        if ch is None:
            raise RuntimeError("must specify either gen_ch or ro_ch!")
        return self.deg2int(deg, ch)

    def reg2deg(self, r, gen_ch=0, ro_ch=None):
        ch = self._get_ch_cfg(gen_ch, ro_ch)
        if ch is None:
            raise RuntimeError("must specify either gen_ch or ro_ch!")
        return self.int2deg(r, ch)

    def _clock(self, gen_ch=None, ro_ch=None):
        ch = self._get_ch_cfg(gen_ch, ro_ch)
        if gen_ch is not None:
            return ch["f_fabric"]
        if ro_ch is not None:
            return ch["f_output"]
        return self["tprocs"][0]["f_time"]

    def cycles2us(self, cycles, gen_ch=None, ro_ch=None):
        return cycles / self._clock(gen_ch, ro_ch)

    def us2cycles(self, us, gen_ch=None, ro_ch=None, as_float=False):
        clock = self._clock(gen_ch, ro_ch)
        return us * clock if as_float else self._to_int(us, clock, parname="length")

    def calc_mixer_freq(self, gen_ch, mixer_freq, nqz, ro_ch):
        gen = self["gens"][gen_ch]
        channels = [self._get_mixer_cfg(gen)]
        if ro_ch is not None:
            channels.append(self["readouts"][ro_ch])
        rounded = self.roundfreq(mixer_freq, channels)
        setval = -rounded if abs(rounded) > gen["fs"] / 2 and nqz == 2 else rounded
        return {"userval": mixer_freq, "rounded": rounded, "setval": setval}

    def calc_muxgen_regs(self, gen_ch, freqs, gains, phases, ro_ch, absolute_freqs, mixer_freq):
        gen = self["gens"][gen_ch]
        channels = [gen] + ([] if ro_ch is None else [self["readouts"][ro_ch]])
        if gains is not None and len(gains) != len(freqs):
            raise RuntimeError("lengths of freqs and gains lists do not match")
        if phases is not None and len(phases) != len(freqs):
            raise RuntimeError("lengths of freqs and phases lists do not match")
        tones = []
        for i, freq in enumerate(freqs):
            rounded = self.roundfreq(freq - mixer_freq if absolute_freqs else freq, channels)
            tone = {"freq_rounded": rounded + mixer_freq if absolute_freqs else rounded}
            if gen["interpolation"] != 1 and abs(rounded) > gen["f_dds"] / 2:
                fs = gen["fs"]
                output = rounded + mixer_freq
                output_zone = int(output / (fs / 2) + 1)
                folded = (-output if output_zone % 2 == 0 else output) % (fs / 2)
                mixer_zone = int(mixer_freq / (fs / 2) + 1)
                image = -folded * (-1) ** mixer_zone + fs * (mixer_zone // 2) - mixer_freq
                if abs(image) <= gen["f_dds"]:
                    rounded = image
            tone["freq_int"] = self.freq2reg(rounded, gen_ch=gen_ch)
            if gen["has_gain"]:
                gain = 1.0 if gains is None else gains[i]
                tone["gain_int"] = int(np.round(gain * gen["maxv"]))
                tone["gain_rounded"] = tone["gain_int"] / gen["maxv"]
            if gen["has_phase"]:
                phase = 0.0 if phases is None else phases[i]
                tone["phase_int"] = self.deg2reg(phase, gen_ch=gen_ch)
                tone["phase_rounded"] = self.reg2deg(tone["phase_int"], gen_ch=gen_ch)
            tones.append(tone)
        return tones

    def calc_ro_regs(self, rocfg, phase, sel, ch=None):
        regs = {}
        if rocfg["has_outsel"]:
            regs["sel"] = sel
        elif sel != "product":
            raise RuntimeError("output selection is not supported by readout %s" % ch)
        if "b_phase" in rocfg:
            regs["phase_int"] = self.deg2int(phase, rocfg)
            regs["phase_rounded"] = self.int2deg(regs["phase_int"], rocfg)
        elif phase != 0:
            raise RuntimeError("phase selection is not supported by readout %s" % ch)
        return regs

    def calc_ro_freq(self, rocfg, ro_pars, ro_regs, absolute_freqs, mixer_freq, flip_freq=False):
        gen_ch = ro_pars["gen_ch"]
        if gen_ch is None:
            freq = self.roundfreq(ro_pars["freq"], [rocfg])
        else:
            mixer = self.roundfreq(mixer_freq, [rocfg])
            dds = ro_pars["freq"] - mixer if absolute_freqs else ro_pars["freq"]
            freq = self.roundfreq(dds, [self["gens"][gen_ch], rocfg]) + mixer
        ro_regs["f_rounded"] = -freq if flip_freq else freq
        if "pfb_nout" in rocfg:
            self._calc_pfbro_freq(rocfg, ro_regs)
        else:
            ro_regs["f_int"] = self.freq2int(ro_regs["f_rounded"], rocfg)
        if flip_freq:
            ro_regs["f_rounded"] *= -1

    def _calc_pfbro_freq(self, rocfg, ro_regs):
        freq = ro_regs["f_rounded"]
        zone = int(freq // (rocfg["fs"] / 2)) + 1
        if zone % 2 == 0:
            freq *= -1
        freq %= rocfg["fs"]
        ro_regs["f_folded"] = freq
        step = rocfg["f_dds"] / 2
        steps = int(np.round(freq / step))
        dds = freq - steps * step
        ro_regs.update({"pfb_f": dds, "pfb_lo": (steps - 0.5) * step,
                        "pfb_hi": (steps + 0.5) * step, "pfb_center": steps * step,
                        "pfb_lolo": max(steps - 1, 0) * step,
                        "pfb_hihi": min(steps + 1, rocfg["pfb_nch"]) * step,
                        "pfb_ch": (rocfg["pfb_ch_offset"] + steps) % rocfg["pfb_nch"],
                        "f_int": self.freq2int(dds, rocfg)})

    def check_pfb_collisions(self, rocfg, cfg1, cfgs):
        logger = logging.getLogger(__name__)
        for cfg2 in cfgs:
            if cfg2["f_rounded"] == cfg1["f_rounded"]:
                continue
            if cfg2["pfb_ch"] == cfg1["pfb_ch"]:
                message = "Readouts at %.3f and %.3f MHz use the same PFB channel." % (cfg1["f_rounded"], cfg2["f_rounded"])
                if rocfg["pfb_dds_on_output"]:
                    logger.warning(message + " Crosstalk is possible.")
                else:
                    raise RuntimeError(message)
            else:
                if abs(cfg2["f_folded"] - cfg1["pfb_center"]) < rocfg["f_dds"] / 2:
                    logger.warning("Readout at %.3f MHz may see crosstalk from %.3f MHz.", cfg1["f_rounded"], cfg2["f_rounded"])
                if abs(cfg1["f_folded"] - cfg2["pfb_center"]) < rocfg["f_dds"] / 2:
                    logger.warning("Readout at %.3f MHz may see crosstalk from %.3f MHz.", cfg2["f_rounded"], cfg1["f_rounded"])

    def get_maxv(self, gen_ch):
        gen = self["gens"][gen_ch]
        return int(np.floor(gen["maxv"] * gen["maxv_scale"]))

_soccfg, _soc = None, None
π = np.pi

yaml.add_representer(np.ndarray, lambda dumper, array: dumper.represent_sequence('tag:yaml.org,2002:seq', array.tolist(), flow_style=True))
yaml.add_multi_representer(np.generic, lambda dumper, data: dumper.represent_data(data.item()))

def connect(ip, port=9999, timeout=10, takeover=True):
    """Connect to the QuICK TCP server and cache its configuration/client pair."""
    global _soc, _soccfg
    from .client import TCPClient
    soc = TCPClient(ip, port, timeout=timeout, takeover=takeover)
    try:
        soccfg = _SocConfig(soc.get_cfg())
    except BaseException:
        soc.close()
        raise
    _soc, _soccfg = soc, soccfg
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

def feistel_network(i, N, seed): # generate pseudo-random permutation
    half_bits = (N - 1).bit_length() // 2 + (N - 1).bit_length() % 2
    limit = 1 << half_bits
    mask = limit - 1
    def round_function(val, round_idx): # a variation of a 32-bit MurmurHash-style mixer.
        x = (val ^ seed ^ round_idx) * 0x45d9f3b
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x = (x >> 16) ^ x
        return x & mask
    def permute(val):
        left = val >> half_bits
        right = val & mask
        for r in range(5):
            left, right = right, left ^ round_function(right, r)
        return (left << half_bits) | right
    curr = permute(i % N)
    while curr >= N:
        curr = permute(curr)
    return curr

def nested_get(d, key): # key is a list
    for k in key:
        d = d[k]
    return d

class Sweep:
    def __init__(self, config, sweepConfig, random=False, progressBar=True):
        self.config = copy.deepcopy(config) # Copy config, avoid changing the original
        self.sweep = [] # [{ "key", "list", "len" }]
        self.progressBar = progressBar
        self.random = random
        self.total = 1
        self.index = 0 # iteration index, increase in order
        self.seed = np.random.randint(2147483647)
        def parse_sweep(key, d): # key is a list
            for k, v in d.items():
                if isinstance(v, dict):
                    parse_sweep(key + [k], v)
                    continue
                if np.iterable(v) and not isinstance(v, str):
                    self.sweep.append({ "key": key + [k], "list": list(v), "len": len(v) })
                    self.total *= len(v)
                    continue
                nested_get(self.config, key)[k] = v
        try:
            parse_sweep([], sweepConfig)
            self.sweep.reverse()
            if self.total == 0:
                raise("Empty Sweep")
        except:
            raise("Invalid Sweep Config")
    def __iter__(self): # initialze all iterator
        if self.progressBar:
            self.progress = tqdm(total=self.total, mininterval=0, miniters=1, desc='quick.Sweep')
        self.index = 0
        return self
    def __next__(self):
        if self.index > 0 and self.progressBar:
            self.progress.update()
        if self.index == self.total:
            if self.progressBar:
                self.progress.close()
            raise StopIteration
        i = feistel_network(self.index, self.total, self.seed) if self.random else self.index
        for s in self.sweep:
            nested_get(self.config, s["key"][:-1])[s["key"][-1]] = s["list"][i % s["len"]]
            i = i // s["len"]
        self.index += 1
        return self.config

class Saver:
    def __init__(self, title, path, indep_params=[], dep_params=[], params={}):
        self.indep_params = []
        self.dep_params = []
        self.params = params
        self.title = title
        self.path = path
        self.file_name = ''
        self.has_data = False
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
        # Creating file name + path
        self.file_name = self.path + f"/{next_number:05d} - {self.title}"
        self.created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write_yml()
        np.savetxt(self.file_name + '.csv', [], delimiter=',')
    def write_yml(self):
        meta = {
            "created": self.created_time,
            "completed": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if self.has_data else "N/A",
            "title": self.title,
            "independent": self.indep_params,
            "dependent": self.dep_params,
            "parameters": self.params
        }
        save_yaml(self.file_name + ".yml", meta)
    def write_data(self, data):
        self.has_data = True
        with open(self.file_name + '.csv', 'a+') as f:
            np.savetxt(f, data, fmt="%.9e", delimiter=',')
        # self.write_yml() # do not call it every time for efficiency

def dB2gain(dB, ref_gain=1, ref_dB=None):
    return ref_gain * 10 ** (dB / 20) if ref_dB is None else 10 ** ((dB + ref_dB) / 20)

def gain2dB(gain, ref_gain=1, ref_dB=None):
    return 20 * np.log10(gain / ref_gain) if ref_dB is None else 20 * np.log10(gain) - ref_dB

def evalStr(s, var, _var=None):
    return eval(f"f'''{s}'''", _var, var) # How amazing this is!

def safe_wrap(retry=2, timeout=300):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retry + 1):
                try: # func_timeout is dark magic. It calls ctypes.pythonapi to kill a thread
                    return func_timeout(timeout, func, args, kwargs)
                except Exception:
                    if i == retry:
                        raise
        return wrapper
    return decorator

def symmetryCenter(x, y, it=3):
    L = len(y)
    R = int(L / 2)
    Δ = int(R / 2)
    freq = np.fft.rfftfreq(L*2)
    y = (y - np.mean(y)) / np.std(y)
    fft = np.fft.rfft(np.append(y, np.zeros(L)))
    res, d = 0, 0
    for _ in range(it): # dark magic: phase shift in Fourier space to align cross-correlation peak to the center iteratively
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

def iq_scatter(S0s, S1s, c0=None, c1=None, plot=True):
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
    o = (c0 + c1) / 2.0 # origin
    g = o - c0 # project on opposite of g
    S0p = np.real((S0s - o) * np.conj(g)) / np.abs(g)
    S1p = np.real((S1s - o) * np.conj(g)) / np.abs(g)
    S0p_right, S1p_right = S0p[S0p < 0], S1p[S1p > 0]
    S0s_right, S0s_wrong = S0s[S0p < 0], S0s[S0p >= 0]
    S1s_right, S1s_wrong = S1s[S1p > 0], S1s[S1p <= 0]
    Fg, Fe = len(S0p_right) / len(S0p), len(S1p_right) / len(S1p)
    phase, threshold = float(-np.angle(c1 - c0, deg=True)), float(np.real((c1 + c0) / 2 * np.exp(-1j *  np.angle(c1 - c0))))
    if not plot:
        return phase, threshold, visibility, Fg, Fe, c0, c1, None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.44)) # plot the histogram
    xp = np.linspace(np.min(S0p), np.max(S1p), 2001) # axis of projection
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
    axr.plot(np.linspace(xp[0], xp[-1], 100), g_cdf - e_cdf, color='green', linewidth=2)
    axes[1].set_xlabel('Projected IQ Amplitude (a.u.)')
    axes[1].set_ylabel('Probability Density')
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
    _r = max(max(np.max(np.real(S0s - o)), np.max(np.real(S1s - o))), max(np.max(np.imag(S0s - o)), np.max(np.imag(S1s - o))))
    axes[0].set_xlim(-1.1 * _r + o.real, 1.1 * _r + o.real)
    axes[0].set_ylim(-1.1 * _r + o.imag, 1.1 * _r + o.imag)
    # plot the cut
    _x = np.linspace(-1.1 * _r + o.real, 1.1 * _r + o.real, 101)
    axes[0].plot(_x, -(c0.real - c1.real) / (c0.imag - c1.imag) * (_x - (c0.real + c1.real) / 2) + (c0.imag + c1.imag) / 2, 'k--')
    # add the circle to the IQ scatter plot
    theta = np.linspace(-π, π, 1001)
    I_g_fit_list = np.median(S0s_right.real) + np.cos(theta) * 3 * g_std
    Q_g_fit_list = np.median(S0s_right.imag) + np.sin(theta) * 3 * g_std
    I_e_fit_list = np.median(S1s_right.real) + np.cos(theta) * 3 * e_std
    Q_e_fit_list = np.median(S1s_right.imag) + np.sin(theta) * 3 * e_std
    axes[0].plot(I_g_fit_list, Q_g_fit_list, color='darkblue', linestyle='--', label=r'$|g\rangle$, $3\sigma$')
    axes[0].plot(I_e_fit_list, Q_e_fit_list, color='darkred', linestyle='--', label=r'$|e\rangle$, $3\sigma$')
    axes[0].legend(shadow=False, loc=1, frameon=True)
    return phase, threshold, visibility, Fg, Fe, c0, c1, fig

# T1 fit and plot
def fitT1(T, S, plot=True):
    def m(x, *p):
        return p[0] * np.exp(-x / p[1]) + p[2]
    def jac(x, *p):
        d0 = np.exp(-x / p[1])
        d1 = p[0] * x / p[1] ** 2 * np.exp(-x / p[1])
        d2 = np.ones(len(x))
        return np.transpose([d0, d1, d2])
    p0 = [1.0, 1.0, 1.0]
    p, pcov = curve_fit(m, T, S, p0=p0, jac=jac)
    perr = np.sqrt(np.diag(pcov))  # Standard deviation of parameters
    residuals = S - m(T, *p)
    r2 = 1 - np.sum(residuals**2) / np.sum((S - np.mean(S))**2)
    if not plot:
        return p, perr, r2, None
    dof = len(T) - len(p)
    fig, ax = plt.subplots()
    ax.scatter(T, S, color='black', s=10, label='Original Data')
    ax.plot(T, m(T, *p), color='red', label='Fit')
    annotation_text = (
        r"$S = S_0e^{-\tau/T_1} + C$" + "\n" +
        "Fitting Result:\n" +
        r"$S_0 = ({:.2f} \pm {:.2f})$".format(p[0], perr[0]) + "\n" +
        r"$T_1 = ({:.2f} \pm {:.2f})$ us".format(p[1], perr[1]) + "\n" +
        r"$C = ({:.2f} \pm {:.2f})$".format(p[2], perr[2]) + "\n" +
        r"$n = {}$ (DOF)".format(dof) + "\n" +
        r"$R^2 = {:.3f}$%".format(r2*100)
    )
    ax.set_xlabel(r"Pulse Delay $\tau$ (us)")
    ax.set_ylabel("Population")
    ax.set_title("T1 Measurement")
    ax.grid()
    ax.legend()
    mid_index = len(T) // 2
    ax.annotate(annotation_text, (T[mid_index], S[mid_index]), fontsize=8, xycoords='data', textcoords='offset points', xytext=(20, 20))
    return p, perr, r2, fig

# T2 fit and plot
def fitT2(T, S, omega=2*π, T2=20.0, plot=True):
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
    p0 = [0.4, T2, omega, 0.5]
    p, pcov = curve_fit(m, T, S, p0=p0, jac=jac)
    perr = np.sqrt(np.diag(pcov))  # Standard deviation of parameters
    residuals = S - m(T, *p)
    r2 = 1 - np.sum(residuals**2) / np.sum((S - np.mean(S))**2)
    if not plot:
        return p, perr, r2, None
    dof = len(T) - len(p)
    fig, ax = plt.subplots()
    ax.scatter(T, S, color='black', s=10, label='Original Data')
    ax.plot(T, m(T, *p), color='red', label='Fit')
    ax.plot(T, me(T, *p), color='blue', label='Fit Exponential')
    annotation_text = (
        r"$S = S_0e^{-\tau/T_2}\cos(\omega\tau) + C$" + "\n" +
        "Fitting Result:\n" +
        r"$S_0 = ({:.2f} \pm {:.2f})$, $C = ({:.2f} \pm {:.2f})$".format(p[0], perr[0], p[3], perr[3]) + "\n" +
        r"$T_2 = ({:.2f} \pm {:.2f})$ us".format(p[1], perr[1]) + "\n" +
        r"$\omega = ({:.2f} \pm {:.2f})$ rad/us".format(p[2], perr[2]) + "\n" +
        r"$n = {}$ (DOF), $R^2 = {:.3f}$%".format(dof, r2 * 100)
    )
    ax.set_xlabel(r"Pulse Delay $\tau$ (us)")
    ax.set_ylabel("Population")
    ax.set_title("T2 Measurement")
    ax.grid()
    ax.legend(loc="lower right")
    mid_index = len(T) // 2
    ax.annotate(annotation_text, (T[mid_index], me(T[mid_index], *p)), fontsize=8, xycoords='data', textcoords='offset points', xytext=(10, 10))
    return p, perr, r2, fig

def fitResonator(F, S, fit="circle", p0=[None, None, None, None], plot=True, normalize_background=True, normalize_phase=True):
    def dB(a):
        return 20. * np.log10(np.abs(a))
    def S21_th(f, *p):
        Qi, Qc, fr, phi = p
        return 1 / (1 + Qi / Qc * np.exp(1j * phi) / (1 + 2j * Qi * (f - fr) / fr))
    def normalize(F, S):
        n = 4
        s = F.argsort()
        F, S = F[s], S[s]
        logmag = dB(S)
        if normalize_background:
            magoffset = np.mean(np.concatenate((logmag[:n], logmag[-n:])))
            logmag = logmag - magoffset
            fit_logmag_background = np.polyfit(np.concatenate((F[:n], F[-n:])), np.concatenate((logmag[:n], logmag[-n:])), 1)
            logmag_background_line = np.poly1d(fit_logmag_background)
            logmag = logmag - logmag_background_line(F)
        phase = np.unwrap(np.angle(S))
        if normalize_phase:
            fitphase = np.polyfit(np.concatenate((F[:n], F[-n:])), np.concatenate((phase[:n], phase[-n:])), 1)
            phaseline = np.poly1d(fitphase)
            phase = phase - phaseline(F)
        S = 10**(logmag/20.)*np.exp(1j*phase)
        return F, S
    F, S = normalize(F, S)
    S_inv, S_dB = 1 / S, dB(S)
    _p0 = [100000, 10000, F[np.argmin(S_dB)], 0.0]
    p0 = list(p0)
    for i in range(len(p0)):
        if p0[i] is None:
            p0[i] = _p0[i]
    bounds = ([-np.inf, -np.inf, 0, -2*π], [np.inf, np.inf, np.inf, 2*π])
    def p_b2i(p):
        _p = []
        for i in range(len(p)):
            if bounds[0][i] == -np.inf:
                _p.append(p[i])
                continue
            if bounds[1][i] == np.inf:
                _p.append(np.sqrt((p[i] - bounds[0][i] + 1) ** 2 - 1))
                continue
            _p.append(np.arcsin(2 * (p[i] - bounds[0][i]) / (bounds[1][i] - bounds[0][i]) - 1))
        return _p
    def p_i2b(_p):
        p = []
        for i in range(len(_p)):
            if bounds[0][i] == -np.inf:
                p.append(_p[i])
                continue
            if bounds[1][i] == np.inf:
                p.append(bounds[0][i] - 1 + np.sqrt(_p[i] ** 2 + 1))
                continue
            p.append(bounds[0][i] + (np.sin(_p[i]) + 1) * (bounds[1][i] - bounds[0][i]) / 2)
        return p
    def err_i2b(_p, _perr):
        perr = []
        for i in range(len(_p)):
            if bounds[1][i] == np.inf:
                perr.append(np.abs(_p[i] / np.sqrt(_p[i]**2 + 1)) * _perr[i])
            else:
                perr.append(np.abs(np.cos(_p[i]) * (bounds[1][i] - bounds[0][i]) / 2) * _perr[i])
        return perr
    def S21_circle_fit_err(_p):
        y = 1 / S21_th(F, *p_i2b(_p))
        return np.concatenate((np.real(S_inv - y), np.imag(S_inv - y)))
    def S21_amp_fit_err(_p):
        return dB(S21_th(F, *p_i2b(_p))) - S_dB
    def S21_arg_fit_err(_p):
        return np.angle(S / S21_th(F, *p_i2b(_p)))
    fun = { "circle": S21_circle_fit_err, "amp": S21_amp_fit_err, "arg": S21_arg_fit_err }
    result, cov, _, _, _ = leastsq(fun[fit], p_b2i(p0), maxfev=50000, xtol=1.e-7, ftol=1.e-7, col_deriv=False, gtol=1.e-7, full_output=True)
    p = p_i2b(result)
    perr = err_i2b(result, np.sqrt(np.diag(cov*np.var(fun[fit](p)))))
    S21_fit = S21_th(F, *p)
    residuals = 1/S - 1/S21_fit
    residuals = np.concatenate((np.real(residuals), np.imag(residuals)))
    flat_S_inv = np.concatenate((np.real(S_inv), np.imag(S_inv)))
    r2 = 1 - np.sum(residuals**2) / np.sum((flat_S_inv - np.mean(flat_S_inv))**2)
    p[0], p[1] = abs(p[0]), abs(p[1])
    if not plot:
        return p, perr, r2, None
    dof = len(residuals) - len(p)
    fig = plt.figure(figsize=(12, 6), tight_layout=True)
    axes = [fig.add_subplot(121), fig.add_subplot(222), fig.add_subplot(224)]
    axes[0].scatter((1 / S).real, (1 / S).imag, label="raw", alpha=0.5, linewidths=0.0)
    axes[0].plot((1 / S21_fit).real, (1 / S21_fit).imag, "r-", label="fit")
    axes[0].set_xlabel(r"Re $\widetilde{S}_{21}^{-1}$")
    axes[0].set_ylabel(r"Im $\widetilde{S}_{21}^{-1}$")
    axes[0].grid()
    axes[1].scatter(F, S_dB, alpha=0.5, linewidths=0.0)
    axes[1].plot(F, dB(S21_fit), "r-")
    axes[1].set_ylabel(r"$|S_{21}|$(dB)")
    axes[1].text(0.025, 0.21, r"$Q_i = %.3e$" % p[0] + "\n" + r"        $\pm %.1e$" % perr[0], ha="left", va="bottom", transform=axes[1].transAxes)
    axes[1].text(0.025, 0.05, r"$Q_c = %.3e$" % p[1] + "\n" + r"        $\pm %.1e$" % perr[1], ha="left", va="bottom", transform=axes[1].transAxes)
    axes[1].text(0.7, 0.05, r"$f = %.7e$" % p[2], ha="left", va="bottom", transform=axes[1].transAxes)
    axes[1].grid()
    mean_phase_data, mean_phase_fit = np.mean(np.unwrap(np.angle(S))), np.mean(np.unwrap(np.angle(S21_fit)))
    match = int((mean_phase_fit - mean_phase_data) / (2 * π) + 0.5)
    axes[2].sharex(axes[1])
    axes[2].scatter(F, np.unwrap(np.angle(S)), alpha=0.5, linewidths=0.0)
    axes[2].plot(F, np.unwrap(np.angle(S21_fit)) - match * 2 * π, 'r-')
    axes[2].text(0.025, 0.95, r"n = {} (DOF)".format(dof), ha="left", va="top", transform=axes[2].transAxes)
    axes[2].text(0.025, 0.87, r"$R^2 = {:.3f}$%".format(r2 * 100), ha="left", va="top", transform=axes[2].transAxes)
    axes[2].set_xlim(F[0], F[-1])
    axes[2].set_xlabel(r"Frequency (MHz)")
    axes[2].set_ylabel(r"$arg~S_{21}$")
    axes[2].grid()
    return p, perr, r2, fig
