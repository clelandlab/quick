"""Binary TCP client for the QuICK board server; no QICK or Pyro dependency."""

from collections import OrderedDict
from copy import deepcopy
from numbers import Integral
import hashlib
import json
import socket
import threading
import numpy as np
from .wire import REQUEST, RESPONSE, DATA, DONE, ERROR, send_frame, recv_frame, json_bytes, configure_socket


class RemoteError(RuntimeError):
    """A board rejected a request or failed an acquisition."""

    def __init__(self, code, message):
        self.code = code
        super().__init__('%s: %s' % (code, message))


class TCPClient:
    """One ordered control/data connection, with independent cancellation.

    Each run yields raw batches as (metadata, arrays). Array keys are channel
    numbers written as strings. A completed software round has event='round'.
    close() or closing a partially consumed iterator cancels the active run.
    """

    native_stream = True

    def __init__(self, host, port=9999, timeout=10, cache_size=8, cache_bytes=64 * 1024 * 1024, takeover=True):
        if timeout is None or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('network timeout must be finite positive seconds')
        if type(takeover) is not bool:
            raise TypeError('takeover must be a boolean')
        self.host, self.port = host, int(port)
        self.timeout, self.cache_size = float(timeout), int(cache_size)
        self.cache_bytes = int(cache_bytes)
        self.takeover = takeover
        self._socket = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._active = False
        self._cache = OrderedDict()
        self._cfg = self._firmware_id = self._session_id = None
        self.get_cfg()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _new_socket(self, timeout=None):
        sock = socket.create_connection((self.host, self.port), self.timeout if timeout is None else timeout)
        configure_socket(sock)
        return sock

    def _next_id(self):
        self._request_id = self._request_id % 0xffffffff + 1
        return self._request_id

    @staticmethod
    def _error(meta):
        message = meta.get('message', 'board request failed')
        if meta.get('error') in ('TimeoutError', 'AcquisitionTimeout'):
            raise TimeoutError(message)
        raise RemoteError(meta.get('error', 'BoardError'), message)

    def _receive(self, request_id):
        frame = recv_frame(self._socket)
        kind, received_id, meta, arrays = frame
        if received_id != request_id:
            raise ConnectionError('response request ID does not match')
        if kind == ERROR:
            if arrays or not isinstance(meta.get('error'), str) or not isinstance(meta.get('message'), str):
                raise ConnectionError('invalid error response')
            # The server reports run errors only after releasing hardware.
            # A transport failure remains ambiguous and still needs stop().
            self._active = False
            self._error(meta)
        return kind, meta, arrays

    def _exchange(self, command, arrays=None, **metadata):
        request_id = self._next_id()
        send_frame(self._socket, REQUEST, request_id, dict(command=command, **metadata), arrays)
        kind, meta, arrays = self._receive(request_id)
        if kind != RESPONSE:
            raise ConnectionError('expected a command response')
        return meta, arrays

    def _connect(self):
        if self._socket is not None:
            return
        self._socket = self._new_socket()
        try:
            self._socket.settimeout(max(self.timeout, 35) if self.takeover else self.timeout)
            meta, arrays = self._exchange('config', takeover=self.takeover)
            self._socket.settimeout(self.timeout)
            if arrays or not isinstance(meta.get('cfg'), dict):
                raise ConnectionError('invalid board configuration response')
            if self.takeover and meta.get('takeover_complete') is not True:
                raise ConnectionError('server did not acknowledge experimental-session takeover')
            self._cfg = meta['cfg']
            self._firmware_id = meta['firmware_id']
            self._session_id = meta['session_id']
            self._cache.clear()
        except BaseException:
            self._disconnect()
            raise

    def _disconnect(self):
        sock, self._socket = self._socket, None
        self._cache.clear()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _request(self, command, arrays=None, **metadata):
        with self._lock:
            try:
                self._connect()
                return self._exchange(command, arrays, **metadata)
            except BaseException:
                self._disconnect()
                raise

    def get_cfg(self):
        with self._lock:
            self._connect()
            return deepcopy(self._cfg)

    @property
    def firmware_id(self):
        with self._lock:
            self._connect()
            return self._firmware_id

    def upload(self, plan, arrays):
        """Upload an immutable execution plan, reusing this session's cache."""
        digest = hashlib.sha256(json_bytes(plan))
        prepared = {}
        for key in sorted(arrays):
            value = np.asarray(arrays[key])
            array = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder('<'))
            digest.update(json_bytes([key, array.dtype.str, list(array.shape)]))
            digest.update(memoryview(array).cast('B'))
            prepared[key] = array
        key = digest.hexdigest()
        with self._lock:
            try:
                self._connect()
                if plan.get('firmware_id') != self._firmware_id:
                    raise ValueError('program configuration does not match connected firmware')
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key][0]
                meta, _ = self._exchange('upload', prepared, plan=plan)
                program_id = meta['program_id']
                size = sum(a.nbytes for a in prepared.values())
                if self.cache_size > 0 and size <= self.cache_bytes:
                    frozen = {name: array.copy() for name, array in prepared.items()}
                    self._cache[key] = (program_id, json.loads(json_bytes(plan)), frozen, size)
                    while len(self._cache) > self.cache_size or sum(v[3] for v in self._cache.values()) > self.cache_bytes:
                        self._cache.popitem(last=False)
                return program_id
            except BaseException:
                self._disconnect()
                raise

    def stream_run(self, program_id, mode='accumulated', rounds=1,
                   start_src='internal', timeout=None, load_envelopes=True):
        """Yield DATA batches and round DONE events until the run completes."""
        with self._lock:
            completed = False
            try:
                self._connect()
                # Heartbeats keep healthy external-trigger waits alive without
                # turning a missing peer into an unlimited network read.
                self._socket.settimeout(self.timeout)
                for attempt in range(2):
                    received = False
                    request_id = self._next_id()
                    self._active = True
                    send_frame(self._socket, REQUEST, request_id,
                               dict(command='run', program_id=program_id, mode=mode,
                                    rounds=rounds, start_src=start_src, timeout=timeout,
                                    load_envelopes=load_envelopes))
                    try:
                        while True:
                            kind, meta, arrays = self._receive(request_id)
                            received = True
                            if kind == DATA:
                                if meta.get('kind') == 'heartbeat':
                                    if arrays or set(meta) != {'kind'}:
                                        raise ConnectionError('invalid stream heartbeat')
                                    continue
                                yield meta, arrays
                            elif kind == DONE and meta.get('event') == 'round':
                                if arrays:
                                    raise ConnectionError('round completion cannot contain arrays')
                                yield meta, arrays
                            elif kind == DONE and meta.get('event') == 'run':
                                if arrays or meta.get('rounds') != rounds:
                                    raise ConnectionError('invalid run completion')
                                completed = True
                                break
                            else:
                                raise ConnectionError('unexpected streaming frame')
                    except RemoteError as exc:
                        saved = next((v for v in self._cache.values() if v[0] == program_id), None)
                        if attempt or received or exc.code != 'CacheMissError' or saved is None:
                            raise
                        # Cache misses are rejected before hardware starts; only
                        # this explicit preflight failure is safe to retry.
                        self._exchange('upload', saved[2], plan=saved[1])
                        continue
                    break
            finally:
                if not completed:
                    if self._active:
                        try:
                            self.stop()
                        except Exception:
                            pass
                    self._disconnect()
                self._active = False
                if self._socket is not None:
                    self._socket.settimeout(self.timeout)

    def stop(self):
        """Cancel this client's run through a separate control connection."""
        if self._session_id is None:
            return False
        with self._new_socket(timeout=min(self.timeout or 2, 2)) as sock:
            send_frame(sock, REQUEST, 1, dict(command='stop', session_id=self._session_id))
            kind, request_id, meta, arrays = recv_frame(sock)
            if kind == ERROR:
                self._error(meta)
            if kind != RESPONSE or request_id != 1 or arrays:
                raise ConnectionError('invalid stop response')
            return bool(meta.get('stopped'))

    def status(self):
        return self._request('status')[0]

    def read_mem(self, length, mem_sel='dmem', addr=0):
        for name, value in (('length', length), ('addr', addr)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError('%s must be an integer' % name)
        return self._request('read_mem', length=int(length), mem_sel=mem_sel, addr=int(addr))[1]['data']

    def reset_gens(self):
        return self._request('reset_gens')[0]

    def close(self):
        if self._active:
            try:
                self.stop()
            except Exception:
                pass
        self._disconnect()
