"""Read the supported ZCU216 AXI topology directly from Vivado's HWH file."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import re
import xml.etree.ElementTree as ET


class Firmware:
    """Hardware descriptions and wiring; no QICK or PYNQ import is needed."""
    GENERATORS = {'axis_signal_gen_v6', 'axis_sg_int4_v2', 'axis_sg_mixmux8_v1'}
    READOUTS = {'axis_dyn_readout_v1', 'axis_pfb_readout_v4'}

    def __init__(self, bitfile):
        self.bitfile = Path(bitfile).resolve()
        self.hwhfile = self.bitfile.with_suffix('.hwh')
        self.root = ET.parse(str(self.hwhfile)).getroot()
        self.modules, self.buses, self.signals = {}, defaultdict(list), defaultdict(list)
        for m in self.root.findall('./MODULES/MODULE'):
            name = m.get('FULLNAME').lstrip('/')
            self.modules[name] = dict(type=m.get('MODTYPE'), version=m.get('HWVERSION'),
                revision=int(m.get('COREREVISION', 0)),
                params={p.get('NAME'): p.get('VALUE') for p in m.findall('./PARAMETERS/PARAMETER')},
                ports={p.get('NAME'): p.attrib for p in m.findall('./PORTS/PORT')},
                buses={p.get('NAME'): p.get('BUSNAME') for p in m.findall('./BUSINTERFACES/BUSINTERFACE')})
            for port, bus in self.modules[name]['buses'].items():
                if bus != '__NOC__': self.buses[bus].append((name, port))
            for port, desc in self.modules[name]['ports'].items():
                self.signals[desc.get('SIGNAME')].append((name, port))
        for p in self.root.findall('./EXTERNALPORTS/PORT'):
            self.signals[p.get('SIGNAME')].append(('', p.get('NAME')))
        self.tproc = self.single('qick_processor')
        self.rfpath = self.single('usp_rf_data_converter')
        if self.modules[self.tproc]['revision'] not in range(21, 28):
            raise ValueError('Unsupported tProc v2 revision')
        self.fingerprint = self.sha256(self.bitfile) + ':' + self.sha256(self.hwhfile)

    @staticmethod
    def sha256(path):
        result = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''): result.update(block)
        return result.hexdigest()

    def single(self, kind):
        names = [n for n, m in self.modules.items() if m['type'] == kind]
        if len(names) != 1: raise ValueError('Expected one %s, found %s' % (kind, names))
        return names[0]

    def peer(self, name, port):
        bus = self.modules[name]['buses'][port]
        peers = [x for x in self.buses[bus] if x != (name, port)]
        if len(peers) != 1: raise ValueError('Ambiguous AXIS connection %s/%s' % (name, port))
        return peers[0]

    def trace(self, name, port, kinds, backward=True):
        seen = set()
        while (name, port) not in seen:
            seen.add((name, port))
            name, port = self.peer(name, port)
            m = self.modules[name]
            if m['type'] in kinds: return name, port
            if m['type'] == 'axis_cdcsync_v1':
                port = ('s' if backward else 'm') + port[1:]
            else:
                prefix = 's' if backward else 'm'
                ports = [p for p in m['buses'] if p.lower().startswith(prefix) and 'axis' in p.lower()]
                if len(ports) != 1:
                    raise ValueError('Unsupported AXIS route through %s' % name)
                port = ports[0]
        raise ValueError('Cycle in AXIS graph')

    def controller(self, name, port):
        name, port = self.trace(name, port, {'qick_processor', 'axis_tmux_v1'})
        cfg = {}
        if self.modules[name]['type'] == 'axis_tmux_v1':
            cfg['tmux_ch'] = int(re.search(r'\d+', port).group())
            name, port = self.trace(name, 's_axis', {'qick_processor'})
        cfg['tproc_ch'] = int(re.search(r'\d+', port).group())
        return cfg

    def dma_route(self, name, port, backward):
        name, port = self.trace(name, port, {'axi_dma', 'axis_switch'}, backward)
        switch, channel = None, None
        if self.modules[name]['type'] == 'axis_switch':
            switch, channel = name, int(re.search(r'\d+', port).group())
            name, port = self.trace(name, 'S00_AXIS' if backward else 'M00_AXIS', {'axi_dma'}, backward)
        return name, switch, channel

    def clock(self, name, port, rfchannels):
        desc = self.modules[name]['ports'][port]
        nominal = float(desc['CLKFREQUENCY']) / 1e6
        signal = desc['SIGNAME']
        # A clock wizard changes the nominal rate but inherits its input scale.
        for _ in range(10):
            drivers = [(n, p) for n, p in self.signals[signal] if n and self.modules[n]['ports'][p].get('DIR') == 'O']
            if len(drivers) != 1: break
            n, p = drivers[0]
            if n == self.rfpath:
                match = re.fullmatch(r'clk_(adc|dac)(\d)', p)
                if match:
                    kind, tile = match.group(1), int(match.group(2))
                    channel = next(c for k, c in rfchannels[kind].items() if int(k[0]) == tile)
                    pars = self.modules[self.rfpath]['params']
                    base = float(pars['C_%s%d_Sampling_Rate' % (kind.upper(), tile)]) * 1000
                    scale = Fraction(channel['fs']/base).limit_denominator(100000)
                    return nominal * float(scale)
            inputs = [v for v in self.modules[n]['ports'].values() if v.get('DIR') == 'I' and v.get('SIGIS') == 'clk']
            if len(inputs) != 1: break
            signal = inputs[0]['SIGNAME']
        return nominal

    def configuration(self, rf=None):
        """Build compiler config from topology and the actual RFDC PLL settings."""
        pars = self.modules[self.rfpath]['params']
        channels, refs = {'dac': {}, 'adc': {}}, set()
        for kind in ('dac', 'adc'):
            for tile in range(4):
                prefix = 'C_%s%d_' % (kind.upper(), tile)
                if pars.get(prefix+'Enable') != '1': continue
                f_ref = float(pars[prefix+'Refclk_Freq'])
                nominal_fs = float(pars[prefix+'Sampling_Rate']) * 1000
                fabric_div = int(round(nominal_fs / float(pars[prefix+'Fabric_Freq'])))
                ratio = Fraction(nominal_fs/f_ref).limit_denominator(10000)
                mult, div = ratio.numerator, ratio.denominator
                hw_tile = None if rf is None else getattr(rf, kind+'_tiles')[tile]
                if hw_tile is not None:
                    pll = hw_tile.PLLConfig
                    f_ref = pll['RefClkFreq']
                    mult, div = pll['FeedbackDivider'], pll['RefClkDivider']*pll['OutputDivider']
                fs = f_ref * mult / div
                refs.add(f_ref)
                for block in range(4):
                    key = '%d%d' % (tile, block)
                    if pars.get('C_%s_Slice%s_Enable' % (kind.upper(), key)) != 'true': continue
                    parameter = 'Interpolation_Mode' if kind == 'dac' else 'Decimation_Mode'
                    factor = int(pars['C_%s_%s%s' % (kind.upper(), parameter, key)])
                    if hw_tile is not None:
                        factor = int(getattr(hw_tile.blocks[block], 'InterpolationFactor' if kind == 'dac' else 'DecimationFactor'))
                    channels[kind][key] = dict(fs=float(fs), fs_mult=int(mult), fs_div=int(div),
                        f_fabric=float(fs/fabric_div), **({'interpolation': factor} if kind == 'dac' else {'decimation': factor}))
        if len(refs) != 1: raise ValueError('ZCU216 needs one common RF reference frequency')
        tmod = self.modules[self.tproc]
        tp = tmod['params']
        tcfg = dict(type='qick_processor', fullpath=self.tproc, revision=tmod['revision'],
                    version=tmod['version'], f_time=self.clock(self.tproc, 't_clk_i', channels),
                    f_core=self.clock(self.tproc, 'c_clk_i', channels),
                    output_pins=[], start_pin=None, stop_pin=None)
        for key in ('pmem', 'dmem', 'wmem'): tcfg[key+'_size'] = 2**int(tp[key.upper()+'_AW'])
        tcfg['dreg_qty'] = 2**int(tp['REG_AW'])
        tcfg['fifo_depth'] = 2**int(tp['FIFO_DEPTH'])
        for port, desc in tmod['ports'].items():
            external = [p for n, p in self.signals.get(desc.get('SIGNAME'), []) if not n]
            if re.fullmatch(r'trig_\d+_o', port) and external:
                tcfg['output_pins'].append(['trig', int(port.split('_')[1]), 0, external[0]])
            if port in ('proc_start_i', 'proc_stop_i') and external:
                tcfg['start_pin' if port == 'proc_start_i' else 'stop_pin'] = external[0]
        tcfg['output_pins'].sort(key=lambda p: p[1])
        gens, ros, routes = [], [], {'gens': {}, 'readouts': {}}
        for name, m in self.modules.items():
            kind, p = m['type'], m['params']
            if kind not in self.GENERATORS: continue
            _, port = self.trace(name, 'm_axis', {'usp_rf_data_converter'}, False)
            dac = port[1:3]
            gc = dict(type=kind, fullpath=name, dac=dac, revision=m['revision'], version=m['version'],
                      maxv=32766, has_dds=True, b_dds=32, b_phase=32,
                      has_mixer=kind != 'axis_signal_gen_v6', **channels['dac'][dac])
            gc.update(self.controller(name, 's_axis' if kind == 'axis_sg_mixmux8_v1' else 's1_axis'))
            gc.update(f_dds=gc['fs']/gc['interpolation'], fdds_div=gc['fs_div']*gc['interpolation'])
            if kind == 'axis_sg_mixmux8_v1': gc.update(n_tones=8, has_gain=True, has_phase=True)
            else:
                samps = 16 if kind == 'axis_signal_gen_v6' else 1
                gc.update(maxlen=2**int(p['N'])*samps, samps_per_clk=samps, complex_env=True,
                          maxv_scale=1.0 if samps == 16 else 0.9)
                routes['gens'][name] = self.dma_route(name, 's0_axis', True)
            gens.append(gc)
        gens.sort(key=lambda g: (g['tproc_ch'], g.get('tmux_ch', -1)))
        for name, m in self.modules.items():
            if m['type'] != 'axis_avg_buffer': continue
            if m['version'] != '1.2': raise ValueError('Only axis_avg_buffer 1.2 is supported')
            rname, port = self.trace(name, 's_axis', self.READOUTS)
            rm = self.modules[rname]
            dyn = rm['type'] == 'axis_dyn_readout_v1'
            _, adcport = self.trace(rname, 's1_axis' if dyn else 's_axis', {'usp_rf_data_converter'})
            adc = adcport[1:3]
            rc = dict(adc=adc, avg_maxlen=2**int(m['params']['N_AVG']), buf_maxlen=2**int(m['params']['N_BUF']),
                has_edge_counter=True, has_weights=False, b_phase=32, b_dds=32,
                has_outsel=dyn, iq_offset=0.0 if dyn else -0.5,
                avgbuf_type=m['type'], avgbuf_fullpath=name, avgbuf_revision=m['revision'], avgbuf_version=m['version'],
                ro_type=rm['type'], ro_fullpath=rname, ro_revision=rm['revision'], ro_version=rm['version'], **channels['adc'][adc])
            trigger = self.modules[name]['ports']['trigger']['SIGNAME']
            peers = [(n, p) for n, p in self.signals[trigger] if n == self.tproc]
            if len(peers) != 1 or not peers[0][1].startswith('trig_'): raise ValueError('Unsupported readout trigger wiring')
            rc.update(trigger_type='tport', trigger_port=int(peers[0][1].split('_')[1]), trigger_bit=0)
            _, feed = self.trace(name, 'm2_axis', {'qick_processor'}, False)
            rc['tproc_ch'] = int(re.search(r'\d+', feed).group())
            down = 8 if dyn else int(rm['params']['N'])//2
            if dyn:
                ctl = self.controller(rname, 's0_axis')
                rc['tproc_ctrl'] = ctl.pop('tproc_ch')
                rc.update(ctl)
            else:
                rc.update(pfb_port=int(re.search(r'\d+', port).group()), pfb_nch=down*2,
                          pfb_nout=8, pfb_ch_offset=down, pfb_dds_on_output=True)
            rc.update(f_dds=rc['fs']/rc['decimation']/(1 if dyn else down),
                      fdds_div=rc['fs_div']*rc['decimation']*(1 if dyn else down),
                      f_output=rc['fs']/(rc['decimation']*down))
            avg = self.dma_route(name, 'm0_axis', False)
            buf = self.dma_route(name, 'm1_axis', False)
            if avg[2] != buf[2]: raise ValueError('Readout switches disagree on channel numbering')
            routes['readouts'][name] = {'avg': avg, 'buf': buf}
            ros.append(rc)
        ros.sort(key=lambda r: routes['readouts'][r['avgbuf_fullpath']]['avg'][2])
        routes['tproc'] = self.dma_route(self.tproc, 's_dma_axis_i', True)[0]
        cfg = dict(board='ZCU216', sw_version='quick-server-1', firmware_id=self.fingerprint,
                   fw_timestamp=self.root.get('TIMESTAMP'), refclk_freq=refs.pop(), tprocs=[tcfg],
                   gens=gens, readouts=ros, iqs=[], time_taggers=[], extra_description=[])
        if not gens or not ros: raise ValueError('Firmware has no supported generators/readouts')
        return cfg, routes
