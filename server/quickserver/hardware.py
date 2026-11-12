"""Small ZCU216 register/DMA backend; requires PYNQ, never the QICK package.

Register layouts and hardware workarounds follow QICK 0.2.418.
Derived portions Copyright (c) Open Quantum Hardware, MIT License:

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
from copy import deepcopy
from time import monotonic, sleep
import os
import numpy as np
from .metadata import Firmware


class BoardHardware:
    """Own the FPGA and its DMA channels in one server process.

    download=False attaches to an already configured matching bitstream. The
    caller must stop other hardware-owning services before either mode is used.
    Sample rates are MHz; DMA operations have a finite timeout in seconds.
    """
    def __init__(self, bitfile, download=True, dac_sample_rates=None,
                 adc_sample_rates=None, dma_timeout=5.0, force_init_clks=False):
        from pynq import Overlay, MMIO, allocate
        from pynq.lib.dma import DMA
        import xrfdc
        if os.environ.get('BOARD', 'ZCU216') != 'ZCU216':
            raise ValueError('This backend supports ZCU216 only')
        self.firmware = Firmware(bitfile)
        self.dma_timeout = float(dma_timeout)
        if self.dma_timeout <= 0: raise ValueError('dma_timeout must be positive')
        self.overlay = Overlay(str(self.firmware.bitfile), download=download, ignore_version=True)
        self.rf = xrfdc.RFdc(self.overlay.ip_dict[self.firmware.rfpath])
        nominal, _ = self.firmware.configuration()
        if force_init_clks or not self.clocks_locked():
            if not download:
                raise RuntimeError('RFDC clocks are unlocked; initialize with download=True')
            import xrfclk
            ref = nominal['refclk_freq']
            xrfclk.set_ref_clks(lmk_freq=ref, lmx_freq=ref*2)
            sleep(1)
            self.overlay.download()
            self.rf = xrfdc.RFdc(self.overlay.ip_dict[self.firmware.rfpath])
        if not self.clocks_locked(): raise RuntimeError('RFDC PLLs are not locked')
        for kind, changes in (('dac', dac_sample_rates), ('adc', adc_sample_rates)):
            for tile, frequency in (changes or {}).items():
                tile = int(tile)
                p = self.firmware.modules[self.firmware.rfpath]['params']
                prefix = 'C_%s%d_' % (kind.upper(), tile)
                if p.get(prefix+'Enable') != '1': raise ValueError('Requested RFDC tile is not enabled')
                frequency = float(frequency)
                maximum = float(p[prefix+'Sampling_Rate'])*1000
                # Independent DAC0 can run slower. Other tiles share fabric
                # clocks; changing one independently would invalidate timing.
                if (kind, tile) != ('dac', 0) and abs(frequency-maximum) > 1e-6:
                    raise ValueError('Sample-rate override is supported only for independent DAC tile 0')
                if not 500 <= frequency <= maximum+1e-6:
                    raise ValueError('Sample rate exceeds the firmware clock design')
                getattr(self.rf, kind+'_tiles')[tile].DynamicPLLConfig(
                    source=xrfdc.CLK_SRC_PLL, ref_clk_freq=float(p[prefix+'Refclk_Freq']), samp_rate=frequency)
        self.cfg, self.routes = self.firmware.configuration(self.rf)
        self.mmio = {}
        needed = {self.firmware.tproc}
        needed.update(g['fullpath'] for g in self.cfg['gens'])
        needed.update(r['avgbuf_fullpath'] for r in self.cfg['readouts'])
        needed.update(r['ro_fullpath'] for r in self.cfg['readouts'] if 'pfb_port' in r)
        needed.update(n for n, m in self.firmware.modules.items() if m['type'] == 'axis_switch')
        for name in needed:
            p = self.firmware.modules[name]['params']
            base = int(p['C_BASEADDR'], 0)
            self.mmio[name] = MMIO(base, int(p['C_HIGHADDR'], 0)-base+1)
        dma_names = {self.routes['tproc']}
        dma_names.update(r[0] for r in self.routes['gens'].values())
        dma_names.update(x[0] for r in self.routes['readouts'].values() for x in r.values())
        self.dmas = {name: DMA(self.overlay.ip_dict[name]) for name in dma_names}
        tc = self.cfg['tprocs'][0]
        maxmem = max(tc[key+'_size'] for key in ('pmem', 'wmem', 'dmem'))
        self._mem_write = allocate((maxmem, 8), dtype=np.int32)
        self._mem_read = allocate((maxmem, 8), dtype=np.int32)
        self._env_write = allocate(max(g.get('maxlen', 0) for g in self.cfg['gens']), dtype=np.int32)
        self._avg_read = allocate(max(r['avg_maxlen'] for r in self.cfg['readouts']), dtype=np.int64)
        self._buf_read = allocate(max(r['buf_maxlen'] for r in self.cfg['readouts']), dtype=np.int32)
        self._buffers = [self._mem_write, self._mem_read, self._env_write, self._avg_read, self._buf_read]
        self._dma_failed = False
        self._dma_reset = False
        self._nqz, self._mixers, self.binprog = {}, {}, None
        self._tproc = self.mmio[self.firmware.tproc]
        self.stop_tproc()
        for reg in range(9): self._write(self._tproc, reg, 0)
        for g in self.cfg['gens']:
            block = self.mmio[g['fullpath']]
            if 'maxlen' in g:
                self._write(block, 0, 0)
                self._write(block, 1, 0)
                if g['type'] == 'axis_signal_gen_v6': self._write(block, 2, 10)
            else:
                for reg in range(24): self._write(block, reg, 0)
                self._write(block, 24, 1)
                self._write(block, 24, 0)
        for r in self.cfg['readouts']:
            block = self.mmio[r['avgbuf_fullpath']]
            for reg in (0, 3, 6, 9, 12): self._write(block, reg, 0)

    @staticmethod
    def _write(block, reg, value):
        block.write(int(reg)*4, int(value) & 0xffffffff)

    @staticmethod
    def _read(block, reg):
        return int(block.read(int(reg)*4))

    def _dma(self, channel, buffer, nbytes, start=None):
        """Bound a DMA wait, then let PYNQ finish cache/bookkeeping work."""
        if self._dma_failed: raise RuntimeError('DMA is faulted; restart the server')
        try:
            channel.transfer(buffer, nbytes=int(nbytes))
            if start is not None: start()
            deadline = monotonic() + self.dma_timeout
            while not channel.idle:
                if channel.error:
                    channel.wait()  # raises the driver's detailed DMA exception
                if monotonic() >= deadline:
                    raise TimeoutError('FPGA DMA did not complete; restart the server before reusing it')
                sleep(0.00005)
            channel.wait()
        except Exception:
            self._dma_failed = True
            # AXI DMA's reset bit resets both directions. Bound this too: a
            # failed fabric clock must not turn cancellation into another hang.
            try:
                resets = [(dma.mmio, 0 if dma.sendchannel is not None else 0x30)
                          for dma in self.dmas.values()]
                for mmio, offset in resets: mmio.write(offset, 4)
                deadline = monotonic() + 0.5
                while any(mmio.read(offset) & 4 for mmio, offset in resets):
                    if monotonic() >= deadline: break
                    sleep(0.001)
                else: self._dma_reset = True
            except Exception:
                pass
            raise

    def _select(self, name, master=0, slave=0):
        if name is None: return
        p = self.firmware.modules[name]['params']
        if not 0 <= master < int(p['NUM_MI']) or not 0 <= slave < int(p['NUM_SI']):
            raise ValueError('Invalid AXIS switch route')
        block = self.mmio[name]
        self._write(block, 0, 0)
        for port in range(int(p['NUM_MI'])): self._write(block, 16+port, 0x80000000)
        self._write(block, 16+master, slave)
        self._write(block, 0, 2)

    def get_cfg(self):
        return deepcopy(self.cfg)

    def clocks_locked(self):
        p = self.firmware.modules[self.firmware.rfpath]['params']
        return all(getattr(self.rf, kind+'_tiles')[tile].PLLLockStatus == 2
                   for kind in ('dac', 'adc') for tile in range(4)
                   if p.get('C_%s%d_Enable' % (kind.upper(), tile)) == '1')

    def _dac(self, ch):
        name = self.cfg['gens'][ch]['dac']
        return name, self.rf.dac_tiles[int(name[0])].blocks[int(name[1])]

    def set_nyquist(self, ch, nqz, force=False):
        if nqz not in (1, 2): raise ValueError('Nyquist zone must be 1 or 2')
        name, block = self._dac(ch)
        if name not in self._nqz: self._nqz[name] = block.NyquistZone
        if force or self._nqz[name] != nqz:
            block.NyquistZone = nqz
            self._nqz[name] = nqz

    def set_mixer_freq(self, ch, f, ro_ch=None, phase_reset=True):
        import xrfdc
        if ro_ch is not None: raise ValueError('Send a pre-rounded mixer frequency')
        if not self.cfg['gens'][ch]['has_mixer']: raise ValueError('Generator has no RF mixer')
        name, block = self._dac(ch)
        if name not in self._mixers: self._mixers[name] = block.MixerSettings['Freq']
        if self._mixers[name] == f: return
        settings = dict(block.MixerSettings)
        if settings['MixerType'] != xrfdc.MIXER_TYPE_FINE: raise RuntimeError('RFDC fine mixer is disabled')
        settings.update(EventSource=xrfdc.EVNT_SRC_IMMEDIATE, Freq=float(f), MixerType=xrfdc.MIXER_TYPE_FINE, PhaseOffset=0)
        block.MixerSettings = settings
        block.UpdateEvent(xrfdc.EVENT_MIXER)
        if phase_reset: block.ResetNCOPhase()
        self._mixers[name] = f

    def config_mux_gen(self, ch, tones):
        g = self.cfg['gens'][ch]
        if g.get('n_tones') != 8 or len(tones) > 8: raise ValueError('Invalid mux tone count')
        block = self.mmio[g['fullpath']]
        for i in range(8):
            if i < len(tones):
                for offset, key in ((0, 'freq_int'), (8, 'phase_int'), (16, 'gain_int')):
                    self._write(block, i+offset, tones[i][key])
            else: self._write(block, i+16, 0)
        self._write(block, 24, 1)
        self._write(block, 24, 0)

    def configure_readout(self, ch, ro_regs):
        raise ValueError('This firmware uses tProc-controlled or PFB readouts')

    def config_mux_readout(self, pfbpath, cfgs, sel=None):
        m = self.firmware.modules.get(pfbpath, {})
        if m.get('type') != 'axis_pfb_readout_v4' or sel is not None:
            raise ValueError('Unsupported PFB readout configuration')
        block = self.mmio[pfbpath]
        nch = int(m['params']['N'])
        for cfg in cfgs:
            ch, port = cfg['pfb_ch'], cfg['pfb_port']
            if not 0 <= ch < nch or not 0 <= port < 8: raise ValueError('Invalid PFB channel or output')
            self._write(block, port, ((ch % 8) << 8) + ch//8)
            self._write(block, 8+2*port, cfg['f_int'])
            self._write(block, 9+2*port, cfg['phase_int'])

    def load_envelope(self, ch, data, addr=0):
        g = self.cfg['gens'][ch]
        data = np.ascontiguousarray(data, dtype='<i2')
        if data.ndim != 2 or data.shape[1] != 2 or not len(data): raise ValueError('Envelope must be nonempty (N,2) int16')
        if 'maxlen' not in g or addr < 0 or addr+len(data) > g['maxlen']: raise ValueError('Envelope memory exceeded')
        if len(data) % g['samps_per_clk']: raise ValueError('Envelope must contain whole fabric cycles')
        dma, switch, port = self.routes['gens'][g['fullpath']]
        self._select(switch, master=port)
        np.copyto(self._env_write[:len(data)], data.view('<i4').reshape(-1))
        block = self.mmio[g['fullpath']]
        self._write(block, 0, addr)
        self._write(block, 1, 1)
        try: self._dma(self.dmas[dma].sendchannel, self._env_write, data.nbytes)
        finally: self._write(block, 1, 0)

    def load_weights(self, ch, data, addr=0):
        raise ValueError('This firmware has no weighted readout buffers')

    def _memory(self, mem_sel, length, addr):
        if mem_sel not in ('pmem', 'wmem', 'dmem'): raise ValueError('Invalid tProc memory')
        if length < 1 or addr < 0 or addr+length > self.cfg['tprocs'][0][mem_sel+'_size']:
            raise ValueError('tProc memory transfer exceeds capacity')
        self._write(self._tproc, 2, addr)
        self._write(self._tproc, 3, length)

    def load_mem(self, data, mem_sel='dmem', addr=0):
        data = np.asarray(data).astype('<i4', copy=False)
        if (mem_sel == 'dmem' and data.ndim != 1) or (mem_sel != 'dmem' and (data.ndim != 2 or data.shape[1] != 8)):
            raise ValueError('Invalid tProc memory array shape')
        self._memory(mem_sel, len(data), addr)
        self._mem_write[:len(data)].fill(0)
        np.copyto(self._mem_write[:len(data), 0] if mem_sel == 'dmem' else self._mem_write[:len(data)], data)
        config = self._read(self._tproc, 1) & ~63
        self._write(self._tproc, 1, config | {'pmem': 7, 'dmem': 11, 'wmem': 15}[mem_sel])
        try: self._dma(self.dmas[self.routes['tproc']].sendchannel, self._mem_write, len(data)*32)
        finally: self._write(self._tproc, 1, config)

    def read_mem(self, length, mem_sel='dmem', addr=0):
        self._memory(mem_sel, length, addr)
        config = self._read(self._tproc, 1) & ~63
        self._write(self._tproc, 1, config | {'pmem': 5, 'dmem': 9, 'wmem': 13}[mem_sel])
        try: self._dma(self.dmas[self.routes['tproc']].recvchannel, self._mem_read, length*32)
        finally: self._write(self._tproc, 1, config)
        data = np.array(self._mem_read[:length, :{'pmem': 3, 'wmem': 6, 'dmem': 1}[mem_sel]], copy=True)
        return data.ravel() if mem_sel == 'dmem' else data

    def load_bin_program(self, binprog, load_mem=True):
        # Check all shapes/capacities before the first memory write.
        result = {}
        for key in ('pmem', 'wmem', 'dmem'):
            value = binprog.get(key)
            if value is None:
                if key == 'pmem': raise ValueError('Program memory is required')
                result[key] = None
                continue
            data = np.asarray(value).astype('<i4', copy=True)
            if not len(data) or len(data) > self.cfg['tprocs'][0][key+'_size']:
                raise ValueError('Invalid '+key+' length')
            if (key == 'dmem' and data.ndim != 1) or (key != 'dmem' and (data.ndim != 2 or data.shape[1] != 8)):
                raise ValueError('Invalid '+key+' shape')
            result[key] = data
        self.load_mem(result['pmem'], 'pmem')
        self.binprog = result
        if load_mem: self.reload_mem()

    def reload_mem(self):
        if self.binprog is None: raise RuntimeError('No program loaded')
        for key in ('wmem', 'dmem'):
            if self.binprog[key] is not None: self.load_mem(self.binprog[key], key)

    def config_avg(self, ch, address=0, length=1, edge_counting=False, high_threshold=0, low_threshold=0):
        if length < 1 or length >= 2**32: raise ValueError('Invalid integration length')
        r = self.cfg['readouts'][ch]
        if not 0 <= address < r['avg_maxlen']: raise ValueError('Invalid accumulated buffer address')
        block = self.mmio[r['avgbuf_fullpath']]
        self.enable_buf(ch, False, False)
        for reg, value in ((1, address), (2, length), (12, edge_counting), (13, high_threshold), (14, low_threshold)):
            self._write(block, reg, value)

    def config_buf(self, ch, address=0, length=1):
        r = self.cfg['readouts'][ch]
        if not 0 < length < r['buf_maxlen'] or not 0 <= address < r['buf_maxlen']:
            raise ValueError('Invalid decimated buffer window')
        block = self.mmio[r['avgbuf_fullpath']]
        self.enable_buf(ch, False, False)
        self._write(block, 7, address)
        self._write(block, 8, length)

    def enable_buf(self, ch, enable_avg=True, enable_buf=True):
        block = self.mmio[self.cfg['readouts'][ch]['avgbuf_fullpath']]
        self._write(block, 0, bool(enable_avg))
        self._write(block, 6, bool(enable_buf))

    def _capture(self, ch, address, length, kind):
        r = self.cfg['readouts'][ch]
        maximum = r[kind+'_maxlen']
        if length is None: length = maximum-2
        if not 0 < length <= maximum or not 0 <= address < maximum:
            raise ValueError('Capture transfer exceeds buffer capacity')
        padded = length + length % 2
        if padded >= maximum:
            # DR_LEN is only log2(capacity) bits: an even-padded full buffer
            # would wrap to zero. Read the boundary in two bounded transfers.
            first = maximum-2
            if first <= 0: raise ValueError('Capture buffer is too small')
            return np.concatenate((self._capture(ch, address, first, kind),
                self._capture(ch, (address+first) % maximum, length-first, kind)))
        block = self.mmio[r['avgbuf_fullpath']]
        dma, switch, port = self.routes['readouts'][r['avgbuf_fullpath']][kind]
        self._select(switch, slave=port)
        reg = 3 if kind == 'avg' else 9
        buffer = self._avg_read if kind == 'avg' else self._buf_read
        size = 8 if kind == 'avg' else 4
        self._write(block, reg+1, address)
        self._write(block, reg+2, padded)
        channel = self.dmas[dma].recvchannel
        # Arm DMA before the buffer reader: pre-filling its stalled stream can
        # duplicate samples in the supplied avg_buffer 1.2 firmware.
        try: self._dma(channel, buffer, padded*size, start=lambda: self._write(block, reg, 1))
        finally: self._write(block, reg, 0)
        if channel.transferred != padded*size: raise RuntimeError('Incomplete readout DMA transfer')
        return np.array(buffer[:padded].view('<i4' if kind == 'avg' else '<i2').reshape(-1, 2)[:length], copy=True)

    def get_accumulated(self, ch, address=0, length=None):
        return self._capture(ch, address, length, 'avg')

    def get_decimated(self, ch, address=0, length=None):
        return self._capture(ch, address, length, 'buf')

    def start_src(self, src):
        if src not in ('internal', 'external'): raise ValueError('Invalid start source')
        self.stop_tproc()
        config = self._read(self._tproc, 1) & ~(1 << 10)
        self._write(self._tproc, 1, config | ((src == 'external') << 10))

    def start_tproc(self):
        if not self._read(self._tproc, 1) & (1 << 10): self._write(self._tproc, 0, 4)

    def stop_tproc(self, lazy=False):
        self._write(self._tproc, 0, 8)

    def clear_tproc_counter(self, addr=1):
        if addr not in (1, 2): raise ValueError('Invalid tProc counter')
        self._write(self._tproc, 0, 64)

    def get_tproc_counter(self, addr=1):
        if addr not in (1, 2): raise ValueError('Invalid tProc counter')
        self._write(self._tproc, 8, 1)
        return self._read(self._tproc, 10+addr)

    def prepare_round(self):
        pass

    def cleanup_round(self):
        pass

    def reset_gens(self):
        """Send zero-gain pulses, including mux masks, without a compiler import."""
        instructions, waves = [0], []
        for i, g in enumerate(self.cfg['gens']):
            conf = (0 if 'n_tones' in g else 9) | (g.get('tmux_ch', 0) << 8)
            raw = (0).to_bytes(12, 'little') + (0).to_bytes(4, 'little') + (3).to_bytes(4, 'little') + conf.to_bytes(2, 'little') + bytes(10)
            waves.append(np.frombuffer(raw, dtype='<i4').tolist())
            address = (i << 6) | g['tproc_ch']
            instructions.append((6 << 69) | (1 << 68) | (3 << 66) | (1 << 63) | (96 << 56) | (address << 39))
        # JUMP HERE after the timed waveform commands have been queued.
        instructions.append((1 << 69) | (1 << 68) | (3 << 66) | (len(instructions) << 45))
        rows = [[x & 0xffffffff, (x >> 32) & 0xffffffff, x >> 64, 0, 0, 0, 0, 0] for x in instructions]
        self.start_src('internal')
        self.clear_tproc_counter()
        self.load_bin_program(dict(pmem=rows, wmem=waves, dmem=None))
        self.start_tproc()
        sleep(0.001)
        self.stop_tproc()

    def close(self):
        self.stop_tproc()
        for ch in range(len(self.cfg['readouts'])): self.enable_buf(ch, False, False)
        # A timed-out DMA may still own these pages; do not recycle them.
        if not self._dma_failed or self._dma_reset:
            for buffer in self._buffers: buffer.freebuffer()
