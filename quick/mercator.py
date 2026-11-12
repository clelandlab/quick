"""Mercator protocols compiled locally for the QuICK TCP server.

The backend follows QICK 0.2.418's register and acquisition conventions.
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

from collections import defaultdict
from fractions import Fraction
from numbers import Number
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm as _Progress
from .helper import dB2gain, _SocConfig

logger = logging.getLogger(__name__)


class _Assembler:
    """Encode Mercator's tProc v2 instructions as 256-bit memory words.

    The processor uses 72 instruction bits; the remaining bits in each memory
    word are zero. This deliberately supports only the instruction forms used
    by Mercator, and rejects unsupported forms rather than silently changing
    their meaning.
    """

    WAIT_TIME_OFFSET = 10
    _conditions = {"1": 0, "Z": 1, "S": 2, "NZ": 3, "NS": 4, "F": 5, "NF": 6, "0": 7}
    _alu = {"+": 0, "-": 2, "AND": 4, "ASR": 6}

    @staticmethod
    def _integer(value, width, signed=False):
        number = int(value)
        low = -(1 << (width - 1)) if signed else 0
        high = (1 << (width - int(signed))) - 1
        if not low <= number <= high:
            raise ValueError(f"tProc value {number} is outside [{low}, {high}]")
        return number & ((1 << width) - 1)

    @classmethod
    def _literal(cls, value, width):
        value = str(value)
        if value.startswith(("#u", "#b", "#h")):
            base = {"u": 10, "b": 2, "h": 16}[value[1]]
            return cls._integer(int(value[2:], base), width)
        if value.startswith(("#", "@", "&")):
            return cls._integer(value[1:], width, signed=value[0] != "&")
        raise ValueError(f"Unsupported tProc literal {value!r}")

    @classmethod
    def _register(cls, name):
        name = str(name)
        if len(name) < 2 or name[0] not in "srw" or not name[1:].isdigit():
            raise ValueError(f"Unsupported tProc register {name!r}")
        kind, index = name[0], int(name[1:])
        if index > {"s": 15, "r": 31, "w": 5}[kind]:
            raise ValueError(f"tProc register {name!r} is out of range")
        return {"s": 0, "r": 32, "w": 64}[kind] + index

    @classmethod
    def _source(cls, inst, full=False):
        """Return the 32 data bits, ALU opcode, and data-format field."""
        if "OP" not in inst:
            return cls._literal(inst["LIT"], 32) if "LIT" in inst else 0, 0, 3
        args = inst["OP"].split()
        if len(args) not in (1, 3):
            raise ValueError(f"Unsupported tProc operation {inst['OP']!r}")
        first = cls._register(args[0]) << 24
        if len(args) == 1:
            return first | (cls._literal(inst["LIT"], 16) if "LIT" in inst else 0), 0, 1
        if args[1] not in cls._alu:
            raise ValueError(f"Unsupported tProc ALU operator {args[1]!r}")
        alu = cls._alu[args[1]]
        if not full:
            alu //= 2
        if args[2].startswith("#"):
            literal = cls._literal(args[2], 24)
            if args[1] == "ASR" and literal > 15:
                raise ValueError("tProc arithmetic shift must be between 0 and 15")
            if "LIT" in inst:
                raise ValueError("A tProc immediate ALU operand cannot share a literal")
            return first | literal, alu, 2
        second = cls._register(args[2]) << 16
        return first | second | (cls._literal(inst["LIT"], 16) if "LIT" in inst else 0), alu, 1

    @classmethod
    def _write(cls, inst):
        if "WR" not in inst:
            return 0, 0
        args = inst["WR"].split()
        if len(args) != 2 or args[1] not in ("op", "imm"):
            raise ValueError(f"Unsupported tProc write option {inst['WR']!r}")
        if ("OP" if args[1] == "op" else "LIT") not in inst:
            raise ValueError(f"Missing source for tProc write option {inst['WR']!r}")
        return 8 | (4 if args[1] == "imm" else 0), cls._register(args[0])

    @classmethod
    def _encode(cls, inst):
        cmd = inst["CMD"]
        try:
            cond = cls._conditions[str(inst.get("IF", "1"))]
        except KeyError:
            raise ValueError(f"Unsupported tProc condition {inst['IF']!r}") from None
        uf = cls._integer(inst.get("UF", 0), 1)
        if cmd == "NOP":
            return 0
        if any(key in inst for key in ("WW", "WP")):
            raise ValueError("Mercator does not support simultaneous wave memory/port writes")
        data, alu, df = cls._source(inst, full=cmd == "REG_WR" and inst.get("SRC") == "op")
        write, dst = cls._write(inst)
        ai = addr = 0
        cfg = (uf << 4) | write | alu
        if cmd == "REG_WR":
            opcode = 4
            if write:
                raise ValueError("REG_WR cannot use a second register write")
            source = inst["SRC"]
            if source == "op":
                if "OP" not in inst:
                    raise ValueError("REG_WR op requires an OP expression")
            elif source in ("imm", "label"):
                if "OP" in inst:
                    raise ValueError("REG_WR immediate cannot use an OP expression")
                data = cls._literal(inst["ADDR"] if source == "label" else inst["LIT"], 32)
                df, cfg = 3, 96 | (uf << 4)
            else:
                raise ValueError(f"Unsupported REG_WR source {source!r}")
            dst = cls._register(inst["DST"])
        elif cmd in ("TEST", "JUMP"):
            opcode = 0 if cmd == "TEST" else 1
            if cmd == "TEST":
                cfg |= 16
            elif inst["ADDR"] == "s15":
                pass
            elif str(inst["ADDR"]).startswith("&"):
                ai, addr = 1, cls._literal(inst["ADDR"], 11) << 6
            else:
                raise ValueError("JUMP address must be an immediate or s15")
        elif cmd == "TIME":
            operations = {"rst": 1, "updt": 2, "set_ref": 4, "inc_ref": 8}
            if inst.get("C_OP") not in operations:
                raise ValueError(f"Unsupported TIME operation {inst.get('C_OP')!r}")
            if any(key in inst for key in ("OP", "WR")) or uf:
                raise ValueError("TIME does not support ALU/register write options")
            opcode, cfg, dst = 2, operations[inst["C_OP"]], 0
            if "LIT" in inst:
                data, df = cls._literal(inst["LIT"], 32), 3
            elif "R1" in inst:
                data, df = cls._register(inst["R1"]) << 16, 1
            elif inst["C_OP"] == "rst":
                data, df = 0, 1
            else:
                raise ValueError("TIME requires a literal or register")
        elif cmd in ("WPORT_WR", "DPORT_WR", "DPORT_RD", "TRIG"):
            opcode = 6
            if "IF" in inst:
                raise ValueError("Port instructions cannot be conditional")
            port = cls._integer(inst["DST"], 5 if cmd == "TRIG" else 6)
            if cmd == "WPORT_WR":
                if inst.get("SRC") != "wmem" or not str(inst.get("ADDR", "")).startswith("&"):
                    raise ValueError("WPORT_WR requires an immediate waveform-memory address")
                ai, cond, cfg = 1, 1, cfg | 64
                addr = cls._literal(inst["ADDR"], 11) << 6
            elif cmd == "DPORT_WR":
                if inst.get("SRC") != "imm":
                    raise ValueError("DPORT_WR requires immediate data")
                ai, cond = 1, 3
                addr = cls._integer(inst["DATA"], 11, signed=True) << 6
            elif cmd == "TRIG":
                if inst.get("SRC") not in ("set", "clr"):
                    raise ValueError("TRIG source must be set or clr")
                ai, cond, port = 1, 3, port + 32
                addr = int(inst["SRC"] == "set") << 6
            else:
                cond = 0
            addr |= port
            if "TIME" in inst:
                if "OP" in inst or "WR" in inst:
                    raise ValueError("Timed port instructions cannot use ALU/register write options")
                data, df, cfg, dst = cls._literal(inst["TIME"], 32), 3, (cfg & 64) | 32, 0
        else:
            raise ValueError(f"Unsupported tProc instruction {cmd!r}")
        # Fields, from most significant to least: opcode/AI/DF/condition,
        # configuration, address, data, destination register.
        return (opcode << 69) | (ai << 68) | (df << 66) | (cond << 63) | (cfg << 56) | (addr << 39) | (data << 7) | dst

    @classmethod
    def list2bin(cls, program_list, label_dict=None):
        """Return readable binary instructions and eight-word memory rows."""
        labels = {} if label_dict is None else label_dict
        binary, rows = [], []
        for original in program_list:
            inst = dict(original)
            inst.setdefault("P_ADDR", len(rows))
            if "LABEL" in inst and "ADDR" not in inst:
                label = inst["LABEL"]
                if label in labels:
                    inst["ADDR"] = labels[label]
                elif label in ("PREV", "HERE", "NEXT", "SKIP"):
                    offset = {"PREV": -1, "HERE": 0, "NEXT": 1, "SKIP": 2}[label]
                    inst["ADDR"] = f"&{inst['P_ADDR'] + offset}"
                else:
                    raise ValueError(f"Unknown tProc label {label!r}")
            if inst["CMD"] == "WAIT":
                if inst.get("C_OP") != "time":
                    raise ValueError("Mercator supports only WAIT time")
                if "ADDR" in inst and inst["ADDR"] != "s15":
                    raise ValueError("An explicit WAIT address must be s15")
                cls._literal(inst["TIME"], 32)
                inst.update(OP=f"s11 - #{int(inst['TIME'][1:]) - cls.WAIT_TIME_OFFSET}", UF="1")
                inst.setdefault("ADDR", f"&{inst['P_ADDR']}")
                expanded = [dict(inst, CMD="TEST"), dict(inst, CMD="JUMP", IF="S")]
            else:
                expanded = [inst]
            for index, item in enumerate(expanded):
                value = cls._encode(item)
                suffix = " //" + original["CMD"] if index == len(expanded) - 1 else ""
                binary.append(f"{value:072b}" + suffix)
                rows.append([value & 0xffffffff, (value >> 32) & 0xffffffff, value >> 64, 0, 0, 0, 0, 0])
        return binary, rows


class _Acquisition:
    """Compile a board execution plan and normalize its acquired IQ arrays."""

    @staticmethod
    def _check_client(soc):
        if getattr(type(soc), 'native_stream', False) is not True:
            raise TypeError('Mercator requires a QuICK TCP client')

    def config_all(self, soc, load_envelopes=True, reset=False, load_mem=True):
        """Prepare the program; TCP applies cached settings atomically at run."""
        self._check_client(soc)
        if self.binprog is None:
            self.compile()
        if reset or not load_mem:
            raise NotImplementedError('TCP config_all stages a complete program; reset=True and load_mem=False are unsupported')
        self._tcp_program_id = soc.upload(*self.to_plan(soc.firmware_id))
        self._tcp_load_envelopes = load_envelopes

    def config_gens(self, soc):
        raise NotImplementedError('TCP generators are configured by config_all() at run time')

    def _readout_settings(self):
        readouts = {}
        pfbs = defaultdict(list)
        for ch, cfg in self.ro_chs.items():
            rocfg = self.soccfg['readouts'][ch]
            if 'tproc_ctrl' in rocfg:
                continue
            gen = self.gen_chs.get(cfg['gen_ch'], {})
            mixer = gen.get('mixer_freq', {}).get('rounded', 0)
            regs = cfg['ro_config']
            self.soccfg.calc_ro_freq(rocfg, cfg, regs, True, mixer, True)
            if 'pfb_port' in rocfg:
                regs['pfb_port'] = rocfg['pfb_port']
                pfb = pfbs[rocfg['ro_fullpath']]
                self.soccfg.check_pfb_collisions(rocfg, regs, pfb)
                pfb.append(regs)
            else:
                readouts[ch] = regs
        muxed = []
        for path, regs in pfbs.items():
            sels = [r.get('sel') for r in regs]
            if len(set(sels)) != 1:
                raise RuntimeError("all declared readouts on a muxed readout must have the same 'sel' setting")
            muxed.append(dict(path=path, regs=regs, sel=sels[0]))
        return readouts, muxed

    def config_readouts(self, soc):
        raise NotImplementedError('TCP readouts are configured by config_all() at run time')

    def to_plan(self, firmware_id=None):
        """Return the versioned execution metadata and binary upload arrays.

        This intermediate format contains hardware registers, tProc memory,
        and DAC envelopes. The board does not need the Mercator compiler.
        """
        from hashlib import sha256
        from .wire import json_bytes
        compiled_for = sha256(json_bytes(self.soccfg.get_cfg())).hexdigest()
        if firmware_id is not None and firmware_id != compiled_for:
            raise ValueError('board configuration changed; reconnect and recompile Mercator')
        readouts, pfbs = self._readout_settings()
        plan = dict(schema=1, firmware_id=compiled_for, counter_addr=self.counter_addr,
                    loop_dims=list(self.loop_dims), generators=[], readouts=[],
                    pfbs=pfbs, envelopes=[], weights=[])
        arrays = {name: np.asarray(data, dtype=np.int64).astype('<i4')
                  for name, data in self.binprog.items() if data is not None}
        for ch, cfg in self.gen_chs.items():
            gen = dict(ch=int(ch), nqz=cfg['nqz'])
            if 'mixer_freq' in cfg:
                gen['mixer_freq'] = cfg['mixer_freq']['setval']
            if 'mux_tones' in cfg:
                gen['mux_tones'] = cfg['mux_tones']
            plan['generators'].append(gen)
        for ch, cfg in self.ro_chs.items():
            ro = {key: cfg[key] for key in ('length', 'trigs', 'edge_counting', 'high_threshold', 'low_threshold')}
            ro['ch'] = int(ch)
            if ch in readouts:
                ro['ro_config'] = readouts[ch]
            plan['readouts'].append(ro)
            rc = self.soccfg['readouts'][ch]
            if rc.get('has_weights', False):
                weights = cfg.get('weights')
                if weights is None:
                    weights = np.zeros((min(cfg['length'], rc['wgt_maxlen']), 2), dtype=np.int16)
                    weights[:, 0] = 2**15 - 2
                name = 'weights_%d' % ch
                arrays[name] = np.ascontiguousarray(weights, dtype='<i2')
                plan['weights'].append(dict(ch=int(ch), array=name))
        for ch, envs in enumerate(self.envelopes):
            blocks = []
            for env in sorted(envs['envs'].values(), key=lambda e: e['addr']):
                if blocks and blocks[-1]['end'] == env['addr']:
                    blocks[-1]['data'].append(env['data'])
                    blocks[-1]['end'] += len(env['data'])
                else:
                    blocks.append(dict(addr=env['addr'], end=env['addr'] + len(env['data']), data=[env['data']]))
            for index, block in enumerate(blocks):
                name = 'env_%d_%d' % (ch, index)
                data = block['data'][0] if len(block['data']) == 1 else np.concatenate(block['data'])
                arrays[name] = np.ascontiguousarray(data, dtype='<i2')
                plan['envelopes'].append(dict(ch=ch, addr=block['addr'], array=name))
        return plan, arrays

    def config_bufs(self, soc, enable_avg=True, enable_buf=True):
        raise NotImplementedError('TCP acquisition buffers are configured by acquire() or acquire_decimated()')

    def load_envelopes(self, soc):
        raise NotImplementedError('TCP envelopes are staged by config_all() and loaded at run time')

    def run(self, soc, load_prog=True, load_envelopes=True, start_src="internal"):
        """Run without collecting data. TCP waits for the program's shot counter."""
        self._check_client(soc)
        try:
            if load_prog:
                self.config_all(soc, load_envelopes=load_envelopes)
            for _ in soc.stream_run(self._tcp_program_id, mode='run', start_src=start_src,
                                    load_envelopes=self._tcp_load_envelopes):
                pass
        except BaseException:
            self._abort_acquire(soc)
            raise

    def setup_counter(self, counter_addr, loop_dims):
        self.counter_addr = counter_addr
        self.loop_dims = loop_dims

    def setup_acquire(self, counter_addr, loop_dims, avg_level):
        self.setup_counter(counter_addr, loop_dims)
        self.avg_level = avg_level

    def set_reads_per_shot(self, reads_per_shot):
        for i, ro in enumerate(self.ro_chs.values()):
            ro['trigs'] = reads_per_shot if isinstance(reads_per_shot, Number) else reads_per_shot[i]

    def get_rounds(self):
        return getattr(self, 'rounds_buf', None)

    def get_raw(self):
        return getattr(self, 'acc_buf', None)

    def get_shots(self):
        return getattr(self, 'shots', None)

    def get_time_axis(self, ro_index, length_only=False):
        ch, ro = list(self.ro_chs.items())[ro_index]
        if length_only:
            return ro['length']
        return self.soccfg.cycles2us(ro_ch=ch, cycles=np.arange(ro['length']))

    def get_time_axis_ddr4(self, ro_ch, data):
        return self.soccfg.cycles2us(ro_ch=ro_ch, cycles=np.arange(data.shape[0]))

    def get_time_axis_mr(self, ro_ch, data):
        return np.arange(data.shape[0]) / self.soccfg['readouts'][ro_ch]['fs']

    def acquire(self, soc, rounds=1, load_envelopes=True, start_src="internal",
                threshold=None, angle=None, progress=True, remove_offset=True,
                step_rounds=False, extra_args=None, timeout=None):
        """Return per-channel IQ arrays, normalized and averaged over hard/soft rounds.

        Each channel has shape (reads, *unaveraged_loops, 2). Thresholding
        returns the fraction above threshold in I and zero in Q. The optional
        timeout is in seconds per round; None waits indefinitely.
        """
        self._validate_acquisition(rounds)
        self.acc_buf = [np.zeros((*self.loop_dims, ro['trigs'], 2), dtype=np.int64)
                        for ro in self.ro_chs.values()]
        self.stats = []
        return self._start_acquire(soc, 'accumulated', rounds, load_envelopes,
                                   start_src, progress, step_rounds, extra_args,
                                   remove_offset=remove_offset, threshold=threshold,
                                   angle=angle, timeout=timeout)

    def acquire_decimated(self, soc, rounds=1, load_envelopes=True,
                          start_src="internal", progress=True, remove_offset=True,
                          step_rounds=False, extra_args=None, timeout=None):
        """Return decimated IQ samples, averaged over software rounds only."""
        self._validate_acquisition(rounds)
        total = math.prod(self.loop_dims)
        for ch, ro in self.ro_chs.items():
            maximum = self.soccfg['readouts'][ch]['buf_maxlen']
            if not 0 < ro['length'] < maximum:
                raise RuntimeError('readout window length must be positive and smaller than buffer capacity')
            if ro['length'] * ro['trigs'] * total > maximum:
                raise RuntimeError("requested readout length (%d x %d trigs x %d reps) exceeds buffer size (%d)"
                                   % (ro['length'], ro['trigs'], total, maximum))
        return self._start_acquire(soc, 'decimated', rounds, load_envelopes,
                                   start_src, progress, step_rounds, extra_args,
                                   remove_offset=remove_offset, timeout=timeout)

    def run_rounds(self, soc, rounds=1, load_envelopes=True, start_src="internal",
                   progress=True, step_rounds=False, timeout=None):
        self._validate_acquisition(rounds, readout=False)
        return self._start_acquire(soc, 'run_rounds', rounds, load_envelopes,
                                   start_src, progress, step_rounds, None,
                                   timeout=timeout)

    def _validate_acquisition(self, rounds, readout=True):
        required = ['counter_addr', 'loop_dims'] + (['avg_level'] if readout else [])
        if any(getattr(self, key, None) is None for key in required):
            raise RuntimeError("data dimensions need to be defined with setup_acquire() before acquisition")
        if not isinstance(rounds, (int, np.integer)) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if any(not isinstance(n, (int, np.integer)) or n < 1 for n in self.loop_dims):
            raise ValueError("loop dimensions must be positive integers")

    def _start_acquire(self, soc, kind, rounds, load_envelopes, start_src,
                       progress, step_rounds, extra_args, **params):
        self._check_client(soc)
        self.acquire_params = dict(type=kind, soc=soc, start_src=start_src,
                                   rounds_remaining=rounds,
                                   hidereps=not (progress and rounds == 1), **params)
        if extra_args is not None:
            self.acquire_params.update(extra_args)
        self.rounds_buf = []
        self._round_prepared = False
        self.rounds_pbar = _Progress(total=rounds,
                                disable=not progress or (rounds == 1 and kind != 'decimated'))
        try:
            self.config_all(soc, load_envelopes=load_envelopes)
            self.prepare_round()
            if step_rounds:
                return None
            self._finish_tcp_rounds(rounds)
            return self.finish_acquire()
        except BaseException:
            self._abort_acquire(soc)
            raise

    def prepare_round(self):
        self._check_client(self.acquire_params['soc'])
        self._round_prepared = True

    def finish_round(self):
        self._check_client(self.acquire_params['soc'])
        self._finish_tcp_rounds(1)
        return self.acquire_params['rounds_remaining'] > 0

    def _finish_tcp_rounds(self, rounds):
        params = self.acquire_params
        soc, kind = params['soc'], params['type']
        total = math.prod(self.loop_dims)
        channels = list(self.ro_chs)
        completed = 0

        if kind == 'run_rounds':
            raw = []
        elif kind == 'accumulated':
            raw = self.acc_buf
        else:
            raw = [np.empty((*self.loop_dims, ro['trigs'], 2), dtype=np.int64)
                   for ro in self.ro_chs.values()]
        dec = [np.empty((total * ro['trigs'] * ro['length'], 2), dtype=np.int16)
               for ro in self.ro_chs.values()] if kind == 'decimated' else []
        raw_positions, dec_positions = {ch: 0 for ch in channels}, {ch: 0 for ch in channels}
        events = soc.stream_run(self._tcp_program_id,
                                mode='run' if kind == 'run_rounds' else kind,
                                rounds=rounds, start_src=params['start_src'],
                                timeout=params.get('timeout'),
                                load_envelopes=self._tcp_load_envelopes)
        try:
            with _Progress(total=total * rounds, disable=params['hidereps']) as pbar:
                for meta, arrays in events:
                    if type(meta.get('round')) is not int or meta['round'] != completed:
                        raise RuntimeError('out-of-order streamed software round')
                    if meta.get('event') == 'round':
                        if type(meta.get('shots')) is not int or meta['shots'] != total:
                            raise RuntimeError('streamed shot count does not match program')
                        if kind != 'run_rounds' and any(n != total for n in raw_positions.values()):
                            raise RuntimeError('incomplete accumulated stream')
                        if kind == 'decimated' and any(dec_positions[ch] != len(dec[i]) for i, ch in enumerate(channels)):
                            raise RuntimeError('incomplete decimated stream')
                        if kind != 'run_rounds':
                            self.acc_buf = raw
                            result = self._process_decimated(dec) if kind == 'decimated' else self._process_accumulated(raw)
                            self.rounds_buf.append(result)
                        completed += 1
                        self._round_prepared = False
                        self.rounds_pbar.update()
                        pbar.update(total)
                        params['rounds_remaining'] -= 1
                        if completed < rounds:
                            raw_positions = {ch: 0 for ch in channels}
                            dec_positions = {ch: 0 for ch in channels}
                        continue
                    frame_kind = meta.get('kind')
                    expected = ('raw', 'decimated') if kind == 'decimated' else ('accumulated',)
                    if kind == 'run_rounds' or frame_kind not in expected:
                        raise RuntimeError('unexpected streamed data kind')
                    selected = meta.get('channels', [])
                    offset, count = meta.get('offset'), meta.get('count')
                    if (not selected or len(set(selected)) != len(selected)
                            or any(type(ch) is not int or ch not in self.ro_chs for ch in selected)
                            or set(arrays) != {str(ch) for ch in selected}
                            or type(offset) is not int or type(count) is not int or offset < 0 or count <= 0):
                        raise RuntimeError('invalid streamed channel or sample range')
                    for ch in selected:
                        index, ro = channels.index(ch), self.ro_chs[ch]
                        positions = dec_positions if frame_kind == 'decimated' else raw_positions
                        factor = 1 if frame_kind == 'decimated' else ro['trigs']
                        target = dec[index] if frame_kind == 'decimated' else raw[index].reshape(-1, 2)
                        data = arrays[str(ch)]
                        if (offset != positions[ch] or (offset + count) * factor > len(target)
                                or data.shape != (count * factor, 2) or data.dtype.kind != 'i'
                                or (frame_kind == 'decimated' and data.dtype.itemsize > 2)):
                            raise RuntimeError('streamed data shape or order does not match program')
                        target[offset * factor:(offset + count) * factor] = data
                        positions[ch] += count
                    if frame_kind == 'accumulated':
                        self.stats.append(meta.get('stats', dict(round=completed, offset=offset, count=count)))
            if completed != rounds:
                raise RuntimeError('stream ended before all software rounds completed')
            if params['rounds_remaining'] <= 0:
                self.rounds_pbar.close()
        except BaseException:
            self._abort_acquire(soc)
            raise
        finally:
            events.close()

    def _abort_acquire(self, soc):
        """Disarm and stop the board without masking the original exception."""
        self._round_prepared = False
        if getattr(soc, '_active', None) is not False:
            try:
                soc.stop()
            except Exception:
                pass
        pbar = getattr(self, 'rounds_pbar', None)
        if pbar is not None:
            pbar.close()

    def finish_acquire(self):
        kind = self.acquire_params['type']
        if kind == 'decimated':
            return self._summarize_decimated(self.rounds_buf)
        if kind == 'accumulated':
            return self._summarize_accumulated(self.rounds_buf)
        return None

    def _process_accumulated(self, acc_buf):
        params = self.acquire_params
        if params['threshold'] is None:
            return self._average_buf(acc_buf, length_norm=True, remove_offset=params['remove_offset'])
        data = [np.zeros_like(d) for d in acc_buf]
        self.shots = self._apply_threshold(acc_buf, params['threshold'], params['angle'], params['remove_offset'])
        for d, shot in zip(data, self.shots):
            d[..., 0] = shot
        return self._average_buf(data, length_norm=False)

    def _summarize_accumulated(self, rounds_buf):
        return [np.mean([r[i] for r in rounds_buf], axis=0) for i in range(len(self.ro_chs))]

    def _ro_offset(self, ch, chcfg):
        rocfg = self.soccfg['readouts'][ch]
        offset = rocfg['iq_offset']
        if chcfg is not None and 'pfb_ch' in chcfg:
            if chcfg['f_int'] == (chcfg['pfb_ch'] % 2) * (2**rocfg['b_dds'] // 2):
                offset *= 2
        return offset

    def _average_buf(self, d_reps, length_norm=True, remove_offset=True):
        result = []
        for i, (ch, ro) in enumerate(self.ro_chs.items()):
            avg = d_reps[i].mean(axis=self.avg_level)
            if length_norm and not ro['edge_counting']:
                avg /= ro['length']
                if remove_offset:
                    avg -= self._ro_offset(ch, ro.get('ro_config'))
            result.append(np.moveaxis(avg, -2, 0))
        return result

    def _apply_threshold(self, acc_buf, threshold, angle, remove_offset):
        thresholds = [float(threshold)] * len(self.ro_chs) if np.ndim(threshold) == 0 else threshold
        angle = 0.0 if angle is None else angle
        angles = [float(angle)] * len(self.ro_chs) if np.ndim(angle) == 0 else angle
        shots = []
        for i, (ch, ro) in enumerate(self.ro_chs.items()):
            avg = acc_buf[i] / ro['length']
            if remove_offset:
                avg -= self.soccfg['readouts'][ch]['iq_offset']
            rotated = np.inner(avg, [np.cos(angles[i]), np.sin(angles[i])])
            shots.append(np.heaviside(rotated - thresholds[i], 0))
        return shots

    def _process_decimated(self, dec_buf):
        total = math.prod(self.loop_dims)
        onetrig = all(ro['trigs'] == 1 for ro in self.ro_chs.values())
        result = []
        for i, (ch, ro) in enumerate(self.ro_chs.items()):
            d = dec_buf[i].astype(float)
            if self.acquire_params['remove_offset']:
                d -= self._ro_offset(ch, ro.get('ro_config'))
            if total != 1 or not onetrig:
                shape = (total * ro['trigs'], -1, 2) if onetrig or total == 1 else (total, ro['trigs'], -1, 2)
                d = d.reshape(shape)
            result.append(d)
        return result

    def _summarize_decimated(self, rounds_buf):
        return [np.mean([r[i] for r in rounds_buf], axis=0) for i in range(len(self.ro_chs))]


class _Program(_Acquisition):
    """Mercator's scalar tProc v2 compiler; hardware access uses a board transport."""
    def __init__(self, soccfg, reps, final_delay, cfg):
        self.soccfg = _SocConfig(soccfg.get_cfg() if hasattr(soccfg, "get_cfg") else soccfg)
        self.tproccfg = self.soccfg["tprocs"][0]
        if self.tproccfg["type"] != "qick_processor" or self.tproccfg["revision"] not in range(21, 28):
            raise RuntimeError("Mercator requires tProc v2 revision 21–27")
        self.cfg = cfg.copy()
        self.gen_chs, self.ro_chs, self.pulses = {}, {}, {}
        self.envelopes = [{"next_addr": 0, "envs": {}} for _ in self.soccfg["gens"]]
        self.waves, self.wave2idx = [], {}
        self._gen_ts = [0] * len(self.soccfg["gens"])
        self._ro_ts = [0] * len(self.soccfg["readouts"])
        self.prog_list, self.labels = [{"CMD": "NOP", "P_ADDR": 0}], {}
        self.p_addr, self.line = 1, 1
        self.loops = [("reps", reps)]
        self.counter_addr, self.avg_level = 1, 0
        self.acc_buf = self.shots = self.rounds_buf = None
        self._asm(CMD="REG_WR", DST="s12", SRC="imm", LIT="#0")
        self._initialize(self.cfg)
        self.delay_auto(1)
        for j, (name, count) in enumerate(self.loops):
            if int(count) != count or count < 1:
                raise ValueError("Loop counts must be positive integers")
            self._asm(CMD="REG_WR", DST=f"r{j}", SRC="imm", LIT="#0")
            self.label(name)
        self._body(self.cfg)
        self.wait_auto(0)
        self.delay_auto(final_delay)
        self._asm(CMD="REG_WR", DST="s12", SRC="op", OP="s12 + #1")
        for j in reversed(range(len(self.loops))):
            name, count = self.loops[j]
            branch = self._branch(name)
            self._asm(CMD="TEST", OP=f"r{j} - #{int(count)-1}")
            self._asm(CMD="JUMP", IF="NZ", WR=f"r{j} op", OP=f"r{j} + #1", **branch)
        self._asm(CMD="JUMP", **self._branch("NEXT" if self._long_jumps else "HERE"))
        self.loop_dims = [int(n) for _, n in self.loops]
        self.binprog = {"pmem": _Assembler.list2bin(self.prog_list, self.labels)[1],
                        "wmem": [self._pack_wave(w) for w in self.waves] or None, "dmem": None}
        for mem, data in self.binprog.items():
            if data is not None and len(data) > self.tproccfg[mem + "_size"]:
                raise RuntimeError(f"Program exceeds {mem} capacity")

    def __getattr__(self, name):
        if name.startswith("_") or "soccfg" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.soccfg, name)

    @property
    def _long_jumps(self):
        return self.tproccfg["pmem_size"] > 2048

    def _asm(self, **inst):
        inst.update(P_ADDR=self.p_addr, LINE=self.line)
        self.prog_list.append(inst)
        self.p_addr += 2 if inst["CMD"] == "WAIT" else 1
        self.line += 1

    def label(self, name):
        if name in self.labels:
            raise RuntimeError(f"Label {name} already defined")
        self.labels[name] = f"&{self.p_addr}"
        self.line += 1

    def _branch(self, name):
        if self._long_jumps:
            self._asm(CMD="REG_WR", DST="s15", SRC="label", LABEL=name)
            return {"ADDR": "s15"}
        return {"LABEL": name}

    def compile(self):
        return self.binprog

    def asm(self):
        return "\n".join(str(i) for i in self.prog_list)

    def add_loop(self, name, count):
        self.loops.append((name, count))

    def declare_gen(self, ch, nqz=1, ro_ch=None, mixer_freq=None, mux_freqs=None, mux_gains=None, mux_phases=None):
        gc = self.soccfg["gens"][ch]
        o = self.gen_chs[ch] = {"nqz": nqz, "ro_ch": ro_ch}
        if gc["has_mixer"]:
            if mixer_freq is None:
                raise ValueError(f"Generator {ch} requires a mixer frequency")
            o["mixer_freq"] = self.soccfg.calc_mixer_freq(ch, mixer_freq, nqz, ro_ch)
        if "n_tones" in gc:
            if mux_freqs is None or len(mux_freqs) > gc["n_tones"]:
                raise ValueError(f"Invalid tone count for generator {ch}")
            o["mux_tones"] = self.soccfg.calc_muxgen_regs(ch, mux_freqs, mux_gains, mux_phases, ro_ch, True, o.get("mixer_freq", {}).get("rounded", 0))

    def declare_readout(self, ch, length, freq=None, phase=0, gen_ch=None):
        rc = self.soccfg["readouts"][ch]
        n = self.us2cycles(length, ro_ch=ch)
        o = self.ro_chs[ch] = {"trigs": 0, "length": n, "length_us": self.cycles2us(n, ro_ch=ch),
                              "edge_counting": False, "high_threshold": 0, "low_threshold": 0}
        if rc.get("has_weights", False):
            o["weights"] = None
        if "tproc_ctrl" not in rc:
            if freq is None:
                raise ValueError(f"Readout {ch} requires a frequency")
            o.update(freq=freq, gen_ch=gen_ch, ro_config=self.soccfg.calc_ro_regs(rc, phase, "product"))

    def add_envelope(self, ch, name, idata=None, qdata=None):
        gc, envs = self.soccfg["gens"][ch], self.envelopes[ch]
        sizes = [len(d) for d in (idata, qdata) if d is not None]
        if not sizes or len(set(sizes)) != 1 or sizes[0] % gc["samps_per_clk"]:
            raise ValueError("Envelope I/Q lengths must match and be whole fabric cycles")
        if not gc.get("complex_env", True) and qdata is not None and np.any(qdata):
            raise ValueError(f"Generator {ch} only supports real envelopes")
        n = sizes[0]
        if envs["next_addr"] + n > gc.get("maxlen", np.inf):
            raise ValueError(f"Generator {ch} envelope memory exceeded")
        data = np.zeros((n, 2), dtype=np.int16)
        for j, d in enumerate((idata, qdata)):
            if d is not None:
                if np.max(np.abs(d)) > gc["maxv"]:
                    raise ValueError("Envelope exceeds DAC range")
                data[:, j] = np.round(d)
        envs["envs"][name] = {"data": data, "addr": envs["next_addr"]}
        envs["next_addr"] += n

    @staticmethod
    def _conf(gc, outsel="product", mode="oneshot", stdysel="zero", phrst=0):
        if outsel is None: outsel = "product"
        if mode is None: mode = "oneshot"
        if stdysel is None: stdysel = "zero"
        if phrst is None: phrst = 0
        return ({"product": 0, "dds": 1, "input": 2, "zero": 3}[outsel]
                + 4 * {"oneshot": 0, "periodic": 1}[mode] + 8 * {"last": 0, "zero": 1}[stdysel]
                + 16 * phrst + (gc.get("tmux_ch") or 0) * 256)

    def _wave(self, name, values):
        if name in self.wave2idx:
            raise ValueError(f"Duplicate waveform {name}")
        self.wave2idx[name] = len(self.waves)
        self.waves.append(tuple(int(x) for x in values))
        return self.wave2idx[name]

    @staticmethod
    def _pack_wave(wave):
        raw = b"".join((int(x) % (1 << (8*n))).to_bytes(n, "little") for x, n in zip(wave, (4, 4, 3, 4, 4, 2)))
        return np.frombuffer(raw[:11] + b"\0" + raw[11:] + bytes(10), dtype="<i4").tolist()

    def add_pulse(self, ch, name, **p):
        ch = [ch] if np.isscalar(ch) else list(ch)
        configs = [self.soccfg["gens"][g] for g in ch]
        if len({(g["type"], g["fs"]) for g in configs}) != 1:
            raise ValueError("Shared pulses require matching generator types and sample rates")
        if "envelope" in p:
            envs = [self.envelopes[g]["envs"][p["envelope"]] for g in ch]
            if len({(e["addr"], len(e["data"])) for e in envs}) != 1:
                raise ValueError("Shared pulse envelopes must have matching lengths and addresses")
        g, gc = ch[0], configs[0]
        waves, dummy = [], False
        if "n_tones" in gc:
            if set(p) != {"style", "mask", "length"} or p["style"] != "const":
                raise ValueError("Multiplexed generators require a constant masked pulse")
            if any(x not in range(len(self.gen_chs[g]["mux_tones"])) for x in p["mask"]):
                raise ValueError("Pulse mask selects an undeclared tone")
            n = self.us2cycles(p["length"], gen_ch=g)
            if not 3 <= n < 2**32:
                raise ValueError("Mux pulse length must be 3 to 2**32-1 cycles")
            mask = 0
            for x in p["mask"]:
                mask |= 1 << x
            waves.append((0, 0, 0, 0, n, mask + (gc.get("tmux_ch") or 0)*256))
        else:
            if p.get("phrst") is not None and gc["type"] not in ("axis_signal_gen_v6", "axis_sg_int4_v1", "axis_sg_int4_v2"):
                raise ValueError(f"Phase reset unsupported for {gc['type']}")
            if not gc.get("has_dds", True) and (p["freq"] != 0 or p["phase"] != 0 or p.get("phrst") == 1):
                raise ValueError("Generator has no DDS")
            f = p["freq"] - self.gen_chs[g].get("mixer_freq", {}).get("rounded", 0)
            if gc["interpolation"] != 1 and abs(f) > gc["f_dds"] / 2:
                raise RuntimeError("requested frequency %f is outside of [-range/2, range/2]" % f)
            rc = None if p.get("ro_ch") is None else self.soccfg["readouts"][p["ro_ch"]]
            f = self.soccfg.freq2int(f, gc, rc)
            phase = self.soccfg._to_int(p["phase"], 2**gc["b_phase"] / 360)
            style = p["style"]
            gain = int(np.trunc(p["gain"] * gc["maxv"] * (gc["maxv_scale"] if style == "const" else 1)))
            if style == "flat_top":
                scale = Fraction(1, 2) * Fraction(gc["maxv_scale"]).limit_denominator(20)
                gain = int(np.trunc(p["gain"]*gc["maxv"]/scale.denominator))*scale.denominator
            if style != "const":
                env = self.envelopes[g]["envs"][p["envelope"]]
                n, addr = len(env["data"])//gc["samps_per_clk"], env["addr"]//gc["samps_per_clk"]
            def segment(length, address=0, amplitude=gain, outsel="product", mode="oneshot", phrst=0):
                if not 3 <= length < 2**16:
                    raise ValueError("Pulse segments must be 3 to 65535 fabric cycles")
                waves.append((f, phase, address, amplitude, length, self._conf(gc, outsel, mode, p.get("stdysel", "zero"), phrst)))
            if style == "const":
                segment(self.us2cycles(p["length"], gen_ch=g), outsel="dds", mode=p.get("mode", "oneshot"), phrst=p.get("phrst", 0))
            elif style == "arb":
                segment(n, addr, mode=p.get("mode", "oneshot"), phrst=p.get("phrst", 0))
            elif style == "flat_top":
                segment(n//2, addr, phrst=p.get("phrst", 0))
                segment(self.us2cycles(p["length"], gen_ch=g), amplitude=int(gain*scale), outsel="dds")
                segment(n//2, addr+(n+1)//2)
                dummy = gc["type"] in ("axis_sg_int4_v1", "axis_sg_int4_v2")
            else:
                raise ValueError(f"Unsupported pulse style {style}")
        if dummy and "dummy" not in self.wave2idx:
            self._wave("dummy", (0, 0, 0, 0, 3, 0))
        indices = [self._wave(f"{name}_w{j}", w) for j, w in enumerate(waves)]
        if dummy:
            indices.append(self.wave2idx["dummy"])
        length = 0
        for j in indices:
            length += self.waves[j][4] / gc["f_fabric"]
        self.pulses[name] = {"waves": indices, "length": length, "chs": ch}

    def add_readoutconfig(self, ch, name, freq, phase=0, gen_ch=None):
        rc = self.soccfg["readouts"][ch]
        mixer = self.gen_chs.get(gen_ch, {}).get("mixer_freq", {}).get("rounded", 0)
        mixer = self.roundfreq(mixer, [rc])
        gc = None if gen_ch is None else self.soccfg["gens"][gen_ch]
        f = -(self.soccfg.freq2int(freq-mixer, rc, gc) + self.freq2reg_adc(mixer, ro_ch=ch))
        phase = self.soccfg._to_int(phase, 2**rc["b_phase"] / 360)
        w = self._wave(f"{name}_w0", (f, phase, 0, 0, 3, self._conf(rc)))
        self.pulses[name] = {"waves": [w]}

    def send_readoutconfig(self, ch, name, t=0):
        self._send(self.soccfg["readouts"][ch]["tproc_ctrl"], self.pulses[name]["waves"], t)

    def _send(self, port, waves, t):
        for w in waves:
            self._asm(CMD="WPORT_WR", DST=str(port), SRC="wmem", ADDR=f"&{w}", TIME=f"@{self.us2cycles(t)}")

    def pulse(self, ch, name, t=0):
        p = self.pulses[name]
        if ch not in p["chs"]:
            raise ValueError("Pulse used on an undeclared generator")
        t = self._gen_ts[ch] if t == "auto" else t
        if t < self._gen_ts[ch]:
            logger.warning("warning: pulse time %s appears to conflict with previous pulse ending at %s?", t, self._gen_ts[ch])
        self._gen_ts[ch] = max(t, self._gen_ts[ch]) + p["length"]
        self._send(self.soccfg["gens"][ch]["tproc_ch"], p["waves"], t)

    def _max_time(self, gens, ros):
        ts = (self._gen_ts if gens else []) + (self._ro_ts if ros else [])
        return max(ts) if ts else None

    def delay(self, t):
        n = self.us2cycles(t)
        if not -2**31 <= n < 2**31:
            raise ValueError("Delay exceeds signed 32-bit time range")
        # Match QICK's parameter rounding, including half-cycle float ties.
        rounded = float(n) * (1 / self.tproccfg["f_time"])
        self._gen_ts = [max(0, x-rounded) for x in self._gen_ts]
        self._ro_ts = [max(0, x-rounded) for x in self._ro_ts]
        self._asm(CMD="TIME", C_OP="inc_ref", LIT=f"#{n}")

    def delay_auto(self, t=0, gens=True, ros=True):
        latest = self._max_time(gens, ros)
        if latest is not None:
            self.delay(latest+t)

    def wait(self, t):
        if t is None:
            return
        n = self.us2cycles(t)
        if -2**23 <= n < 2**23:
            if self._long_jumps:
                self._branch("SKIP")
            self._asm(CMD="WAIT", C_OP="time", TIME=f"@{n}", **({"ADDR": "s15"} if self._long_jumps else {}))
        elif -2**31 <= n < 2**31:
            reg = f"r{len(self.loops)}"
            self._asm(CMD="REG_WR", DST=reg, SRC="imm", LIT=f"#{n-10}")
            if self._long_jumps:
                self._branch("SKIP")
            self._asm(CMD="TEST", OP=f"s11 - {reg}")
            self._asm(CMD="JUMP", OP=f"s11 - {reg}", IF="S", UF="1", **({"ADDR": "s15"} if self._long_jumps else {"LABEL": "HERE"}))
        else:
            raise ValueError("Wait exceeds signed 32-bit time range")

    def wait_auto(self, t=0, gens=False, ros=True):
        latest = self._max_time(gens, ros)
        if latest is not None:
            self.wait(latest+t)

    def trigger(self, ros=None, pins=None, t=0):
        ports, triggers = {}, set()
        for ch in ([] if ros is None else ros):
            rc, ro = self.soccfg["readouts"][ch], self.ro_chs[ch]
            if rc["trigger_type"] == "dport":
                ports[rc["trigger_port"]] = ports.get(rc["trigger_port"], 0) | (1 << rc["trigger_bit"])
            else:
                triggers.add(rc["trigger_port"])
            if t is not None:
                if t < self._ro_ts[ch]:
                    logger.warning("Readout time %d appears to conflict with previous readout ending at %f?", t, self._ro_ts[ch])
                self._ro_ts[ch] = t + ro["length_us"]
            ro["trigs"] += 1
        for pin in ([] if pins is None else pins):
            kind, port, bit, _ = self.tproccfg["output_pins"][pin]
            if kind == "dport":
                ports[port] = ports.get(port, 0) | (1 << bit)
            else:
                triggers.add(port)
        for high in (True, False):
            if not high and t is None:
                self._asm(CMD="REG_WR", DST="s14", SRC="op", OP="s14 + #10")
            timing = {} if t is None else {"TIME": f"@{self.us2cycles(t)+(0 if high else 10)}"}
            for port, bits in ports.items():
                self._asm(CMD="DPORT_WR", DST=str(port), SRC="imm", DATA=str(bits if high else 0), **timing)
            for port in triggers:
                self._asm(CMD="TRIG", DST=str(port), SRC="set" if high else "clr", **timing)

    def read_and_jump(self, ro_ch, component, threshold, test, label):
        self._asm(CMD="DPORT_RD", DST=str(self.soccfg["readouts"][ro_ch]["tproc_ch"]))
        branch = self._branch(label)
        self._asm(CMD="TEST", OP=f"{'s8' if component == 'I' else 's9'} - #{threshold}", UF="1")
        self._asm(CMD="JUMP", IF={"<": "S", ">=": "NS"}[test], **branch)



def generate_waveform(o, soccfg):
    g = o["gs"][-1]
    f_fabric = soccfg["gens"][g]["f_fabric"]
    samps_per_clk = soccfg["gens"][g].get("samps_per_clk", 1)
    l = int((0 if o["length"] is None else o["length"]) * f_fabric) * samps_per_clk
    σ = o["sigma"] * samps_per_clk * f_fabric # keep float!
    if o["style"] == "const":
        o["idata"] = None
    if o["style"] in ["flat_top", "stage"]: # l computed from σ
        l = int(σ * 5 / samps_per_clk / 2) * 2 * samps_per_clk
    if o["style"] in ["gaussian", "flat_top", "DRAG"]:
        x = np.arange(0, l)
        μ = l/2 - 0.5
        y = np.exp(-((x - μ) / σ) ** 2)
        o["idata"] = (y - y[0]) / (1 - y[0])
    if o["style"] == "stage":
        o["idata"] = np.zeros(int(σ * 4))
        for s in o["stage"]:
            o["idata"] = np.append(o["idata"], np.zeros(int(s[1] * f_fabric * samps_per_clk)) + s[0])
        o["idata"] = np.append(o["idata"], np.zeros(int(σ*4)))
        # pad o["idata"] to be integer multiple of samps_per_clk
        o["idata"] = np.append(o["idata"], np.zeros(samps_per_clk - len(o["idata"]) % samps_per_clk))
        o["idata"] = gaussian_filter(o["idata"], σ)
    if o["style"] == "DRAG":
        δ = -o["delta"] / (samps_per_clk * f_fabric)
        o["qdata"] = 0.5 * (x - μ) / (2 * σ ** 2) * o["idata"] / δ
    if o.get("idata") is not None:
        o["idata"] = np.array(o["idata"]) * (1 - o["ibias"]) + o["ibias"]
    if o.get("qdata") is not None:
        o["qdata"] = np.array(o["qdata"]) * (1 - o["qbias"]) + o["qbias"]
    if o["style"] in ["stage", "arb"] and o["length"] is not None: # truncate or pad to length
        l = int(o["length"] * f_fabric) * samps_per_clk # cutoff length
        if o.get("idata") is not None:
            o["idata"] = np.append(o["idata"][:l], np.zeros(max(0, l - len(o["idata"]))))
        if o.get("qdata") is not None:
            o["qdata"] = np.append(o["qdata"][:l], np.zeros(max(0, l - len(o["qdata"]))))
    else:
        o["length"] = 2 if o["length"] is None else o["length"] # default length for non-truncating pulse

def parse(soccfg, cfg):
    """
    Creates the internal control object which controls the pulse sequence.
    c = {
        "p": {} # pulses
        "g": {} # Generator channels
        "r": {} # Readout channels
        "exec": [] # Execution list
    }
    """
    c = { "p": {}, "g": {}, "r": {}, "exec": [] }
    steps = cfg.pop("steps", []) # flatten steps
    for i in range(len(steps)):
        for key in steps[i]:
            if not f"{i}_{key}" in cfg:
                cfg[f"{i}_{key}"] = steps[i][key]
    for i in range(99999999): # find all execution steps
        if not f"{i}_type" in cfg:
            break
        o = { "type": cfg[f"{i}_type"] }
        o["t"] = cfg.get(f"{i}_t", 0)
        if o["type"] == "pulse":
            o["g"], o["p"], o["r"] = cfg.get(f"{i}_g"), cfg.get(f"{i}_p"), cfg.get(f"{i}_r", 0)
            c["g"][o["g"]] = { "p": o["p"], "r": None }
            if o["p"] not in c["p"]:
                c["p"][o["p"]] = { "gs": [o["g"]], "r": None }
            else:
                c["p"][o["p"]]["gs"].append(o["g"])
            o["threshold"] = cfg.get(f"{i}_threshold")
        if o["type"] == "trigger":
            o["rs"], o["p"] = cfg.get(f"{i}_rs"), cfg.get(f"{i}_p")
            if o["rs"] != None:
                for r in o["rs"]:
                    c["r"][r] = {}
        if o["type"] == "pin":
            o["pins"] = cfg.get(f"{i}_pins", [0])
        if o["type"] == "goto":
            o["rep"] = cfg.get(f"{i}_rep", 0)
            o["i"] = cfg.get(f"{i}_i", i + 1)
        c["exec"].append(o)
    for p in c["p"]: # find all used pulses
        o = c["p"][p]
        o["freq"] = cfg.get(f"p{p}_freq", 0)
        for g in o["gs"]:
            c["g"][g]["nqz"] = cfg.get(f"p{p}_nqz", 2)
            c["g"][g]["freq"] = o["freq"]
            c["g"][g]["mixer"] = c["g"][g].get("mixer", cfg.get(f"p{p}_mixer"))
        o["mode"] = cfg.get(f"p{p}_mode", "oneshot")
        o["phrst"] = cfg.get(f"p{p}_phrst", 0)
        o["style"] = cfg.get(f"p{p}_style", "const")
        o["length"] = cfg.get(f"p{p}_length")
        o["sigma"] = cfg.get(f"p{p}_sigma", (0 if o["length"] is None else o["length"]) / 5)
        o["delta"] = cfg.get(f"p{p}_delta", -200)
        o["gain"] = cfg.get(f"p{p}_gain", 0)
        o["mask"] = cfg.get(f"p{p}_mask", list(range(len(o["freq"]))) if np.iterable(o["freq"]) else None)
        o["phase"] = cfg.get(f"p{p}_phase", list(np.zeros(len(o["freq"]))) if np.iterable(o["freq"]) else 0)
        o["stage"] = cfg.get(f"p{p}_stage", [])
        o["idata"] = np.array(cfg[f"p{p}_idata"]) if cfg.get(f"p{p}_idata") is not None else None
        o["qdata"] = np.array(cfg[f"p{p}_qdata"]) if cfg.get(f"p{p}_qdata") is not None else None
        o["ibias"] = cfg.get(f"p{p}_ibias", 0)
        o["qbias"] = cfg.get(f"p{p}_qbias", 0)
        if cfg.get(f"p{p}_power") is not None:
            o["gain"] = dB2gain(cfg[f"p{p}_power"])
            cfg[f"p{p}_gain"] = o["gain"] # write back computed gain
        if o["style"] == "flat_top": # not supporting mode
            o["mode"] = "oneshot"
        generate_waveform(o, soccfg)
    for r in range(10): # find all readout channels
        if f"r{r}_p" in cfg or f"r{r}_freq" in cfg:
            c["r"][r] = o = { "g": None }
            o["p"] = cfg.get(f"r{r}_p")
            o["freq"] = cfg.get(f"r{r}_freq")
            o["length"] = cfg.get(f"r{r}_length")
            o["phase"] = cfg.get(f"r{r}_phase", 0)
            if o["p"] is not None: # match corresponding pulse/channels
                o["g"] = c["p"][o["p"]]["gs"][-1] # use the last generator for matching
                c["g"][o["g"]]["r"] = c["p"][o["p"]]["r"] = r
                if o["freq"] is None:
                    o["freq"] = c["p"][o["p"]]["freq"]
                if o["length"] is None:
                    o["length"] = c["p"][o["p"]].get("length")
            if o["length"] is None:
                o["length"] = 2 # default readout length
    cfg["rep"] = cfg.get("rep", 0)
    return c

class Mercator(_Program):
    """General class for preparing and sending a pulse sequence."""
    def _initialize(self, cfg):
        c = self.c
        mux_gain_factor = { 1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8 }
        for g, o in c["g"].items(): # Declare Generator Channels
            kwargs = {}
            if self.soccfg["gens"][g]["has_mixer"]:
                kwargs["ro_ch"] = o["r"]
                if np.iterable(o["freq"]): # mux
                    kwargs["mixer_freq"] = np.mean(o["freq"]) if o["mixer"] is None else o["mixer"]
                    kwargs["mux_freqs"] = o["freq"]
                    kwargs["mux_gains"] = np.array(c["p"][o["p"]]["gain"]) * mux_gain_factor[len(o["freq"])]
                    kwargs["mux_phases"] = c["p"][o["p"]]["phase"]
                else: # single tone
                    kwargs["mixer_freq"] = o["freq"] if o["mixer"] is None else o["mixer"]
            self.declare_gen(ch=g, nqz=o["nqz"], **kwargs)
        for r, o in c["r"].items(): # Declare Readout Channels
            kwargs = { "phase": o["phase"], "freq": o["freq"] }
            if o.get("g") is not None:
                kwargs["gen_ch"] = o["g"]
            if "tproc_ctrl" in self.soccfg["readouts"][r]:
                self.declare_readout(ch=r, length=o["length"])
                self.add_readoutconfig(ch=r, name=f"r{r}", **kwargs)
                self.send_readoutconfig(ch=r, name=f"r{r}", t=0)
            else: # mux readout
                self.declare_readout(ch=r, length=o["length"], **kwargs)
        for p, o in c["p"].items(): # Setup pulses
            kwargs = { "style": o["style"], "ro_ch": o["r"], "freq": o["freq"], "phase": o["phase"], "gain": o["gain"] }
            if o["mode"] == "last":
                kwargs["stdysel"] = "last"
                kwargs["mode"] = "oneshot"
            else:
                kwargs["mode"] = o["mode"]
            if not np.iterable(o["freq"]): # single tone
                kwargs["phrst"] = o["phrst"]
            if o["style"] == "const" and o["mask"] is not None: # mux mask
                kwargs["mask"] = o["mask"]
                for k in ["freq", "ro_ch", "mode", "phase", "gain"]:
                    kwargs.pop(k, None)
            if o["style"] != "const":
                kwargs["envelope"] = f"e{p}"
                for g in o["gs"]:
                    maxv = self.soccfg.get_maxv(g)
                    idata = None if o["idata"] is None else maxv * np.array(o["idata"])
                    qdata = None if o["qdata"] is None else maxv * np.array(o["qdata"])
                    self.add_envelope(ch=g, name=kwargs["envelope"], idata=idata, qdata=qdata)
            if o["style"] == "flat_top":
                kwargs.pop("mode", None) # no support
            if o["style"] in ["flat_top", "const"]:
                kwargs["length"] = o["length"]
            else:
                kwargs["style"] = "arb"
            self.add_pulse(ch=o["gs"], name=f"p{p}",  **kwargs)
        if cfg["rep"] > 0:
            self.add_loop("rep", cfg["rep"])

    def _body(self, cfg):
        c = self.c
        goto_rep = {} # goto rep
        i = 0
        while i < len(c["exec"]):
            o = c["exec"][i]
            if o["type"] == "pulse":
                if o["threshold"] is not None: # conditional
                    self.read_and_jump(ro_ch=o["r"], component="I", threshold=self.us2cycles(o["threshold"] * self.c["r"][o["r"]]["length"], ro_ch=o["r"]), test="<", label=f"after_{i}")
                self.pulse(ch=o["g"], name=f"p{o['p']}", t=o["t"])
                if o["threshold"] is not None:
                    self.label(f"after_{i}")
            _wait_auto = lambda t: self.wait_auto(t, gens=True) # redefine to include gens
            time_functions = { "delay": self.delay, "delay_auto": self.delay_auto, "wait": self.wait, "wait_auto": _wait_auto }
            if o["type"] in time_functions:
                time_functions[o["type"]](o["t"])
            if o["type"] == "trigger":
                self.trigger(ros=(o["rs"] or self.c["r"].keys()), t=o["t"])
            if o["type"] == "pin":
                self.trigger(pins=o["pins"], t=o["t"])
            if o["type"] == "goto":
                goto_rep[i] = goto_rep.get(i, o["rep"]) - 1
                i = o["i"] if goto_rep[i] >= 0 else i + 1
            else:
                i = i + 1

    def __init__(self, soccfg, cfg):
        self.c = parse(soccfg, cfg)
        super().__init__(soccfg, reps=cfg.get("hard_avg", 1), final_delay=1, cfg=cfg)

    def acquire(self, soc, progress=False, **kwargs):
        res = super().acquire(soc, progress=progress, rounds=self.cfg.get("soft_avg", 1), **kwargs)
        return None if res is None else np.moveaxis(res, -1, 0)

    def acquire_decimated(self, soc, progress=False, **kwargs):
        res = super().acquire_decimated(soc, progress=progress, rounds=self.cfg.get("soft_avg", 1), **kwargs)
        return None if res is None else np.moveaxis(res, -1, 0)

    def light(self):
        fig, ax = plt.subplots()
        c = self.c
        data = {} # plot data
        last = {} # last pulse in each generator
        for g in range(15, -1, -1):
            if g in c["g"]:
                data[g] = [[0, 0]]
                last[g] = { "mode": "oneshot", "end": 0 }
        delays = [0] # for plot, all t are absolute
        r_labeled = {} # for plot
        delay = 0 # current time reference
        wait = 0 # wait until time
        until = 0 # last pulse/readout end time
        def add_pulse(p, g, start, new_pulse=True): # return end time
            cycle = self.cycles2us(1, gen_ch=g)
            us = cycle / self.soccfg["gens"][g]["samps_per_clk"]
            if new_pulse and last[g]["mode"] == "periodic":
                while last[g]["end"] < start:
                    add_pulse(last[g]["p"], g, last[g]["end"] + cycle, new_pulse=False)
            start = max(start, last[g]["end"])
            if last[g]["mode"] == "last":
                data[g].append([start, last[g]["value"]])
            o = c["p"][p]
            last[g] = { "p": p, "mode": o["mode"] }
            gain = o["gain"][0] if np.iterable(o["gain"]) else o["gain"]
            if new_pulse:
                ax.annotate(f"p{p}", (start, gain + 0.05 if gain >= 0 else gain - 0.1))
            if o["idata"] is not None:
                l = o["length"] if o["style"] == "flat_top" else 0
                end = start + l + len(o["idata"]) * us
                h = len(o["idata"]) // 2
                data[g].extend(list(zip(np.arange(h)*us + start, gain * np.array(o["idata"])[:h])))
                data[g].extend(list(zip(np.arange(h)*us + start + l + h*us, gain * np.array(o["idata"])[h:])))
                if o["mode"] == "last":
                    last[g]["value"] = gain * o["idata"][-1]
            else:
                end = start + o["length"]
                if new_pulse:
                    data[g].append([start, 0])
                data[g].extend([[start, gain], [end, gain]])
                if o["mode"] == "last":
                    last[g]["value"] = gain
            if o["mode"] not in ["last", "periodic"]:
                data[g].append([end, 0])
            last[g]["end"] = end
            return end
        goto_rep = {}
        i = 0
        while i < len(c["exec"]):
            o = c["exec"][i]
            start = delay + o["t"]
            end = start
            if o["type"] == "pulse":
                start = max(start, wait, last[o["g"]]["end"])
                end = add_pulse(o["p"], o["g"], start)
                until = max(until, end)
            if o["type"] == "delay":
                delay = delay + o["t"]
                delays.append(delay)
            if o["type"] == "delay_auto":
                delay = max(until, delay) + o["t"]
                delays.append(delay)
            if o["type"] == "wait": # very confusing, but that's the behavior
                wait = delay + o["t"]
            if o["type"] == "wait_auto": # also confusing
                wait = until + o["t"]
            if o["type"] == "trigger":
                for r in (o["rs"] or c["r"]):
                    end = start + c["r"][r]["length"]
                    until = max(end, until)
                    ax.axvspan(xmin=start, xmax=end, color=('r' if r == 0 else 'b'), alpha=0.1, label=(None if r_labeled.get(r) else f"r{r}"))
                    r_labeled[r] = True
            if o["type"] == "goto":
                goto_rep[i] = goto_rep.get(i, o["rep"]) - 1
                i = o["i"] if goto_rep[i] >= 0 else i + 1
            else:
                i = i + 1
        final = max(until, delay)
        for g in data:
            cycle = self.cycles2us(1, gen_ch=g)
            if last[g]["mode"] == "periodic":
                while last[g]["end"] < final:
                    add_pulse(last[g]["p"], g, last[g]["end"] + cycle, new_pulse=False)
            elif last[g]["mode"] == "last":
                data[g].append([final, last[g]["value"]])
            else:
                data[g].append([final, 0])
            xy = np.transpose(data[g])
            ax.plot(xy[0], xy[1], label=f"g{g}", linewidth=3)
        ax.scatter(np.array(delays), np.zeros(len(delays)) - 1, label="delay", marker="^")
        ax.legend()
        ax.set_title("Mercator Light")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Gain")
        ax.set_ylim([-1.05, 1.2])
        ax.grid()
        return fig

# Python ternary operator and object operations are so stupid and makes this code very unreadable. Blame Python syntax here.
