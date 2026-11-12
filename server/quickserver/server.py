"""Typed program upload and bounded acquisition streaming over TCP."""

import argparse
from collections import OrderedDict
import hashlib
import json
import logging
import math
import queue
import select
import signal
import socket
import threading
import time
import uuid
import numpy as np

from .wire import REQUEST, RESPONSE, DATA, DONE, ERROR, MAX_PAYLOAD
from .wire import ProtocolError, json_bytes, recv_frame, send_frame, configure_socket

log = logging.getLogger(__name__)
HEARTBEAT_INTERVAL = 1.0


class BusyError(RuntimeError):
    pass


class CancelledError(RuntimeError):
    pass


class CacheMissError(RuntimeError):
    pass


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid %s" % name)
    return value


class QuickServer:
    """One hardware owner, independent TCP sender, and a bounded program cache."""

    def __init__(self, backend, host="127.0.0.1", port=9999, queue_depth=8,
                 cache_bytes=128 * 1024 * 1024, max_clients=8, heartbeat_interval=HEARTBEAT_INTERVAL):
        self.backend = backend
        _integer(queue_depth, 1, 64, "queue depth")
        _integer(cache_bytes, 1, 512 * 1024 * 1024, "cache size")
        _integer(max_clients, 1, 32, "client limit")
        if not isinstance(heartbeat_interval, (int, float)) or not math.isfinite(heartbeat_interval) or heartbeat_interval <= 0:
            raise ValueError("Heartbeat interval must be positive seconds")
        self.heartbeat_interval = heartbeat_interval
        self.cfg = json.loads(json_bytes(backend.get_cfg()))
        self.firmware_id = hashlib.sha256(json_bytes(self.cfg)).hexdigest()
        self.queue_depth, self.cache_bytes = queue_depth, cache_bytes
        self.cache, self.cache_size = OrderedDict(), 0
        self._state = threading.Lock()
        self._board = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_clients)
        self._cancel_slots = threading.BoundedSemaphore(1)
        self._stop = threading.Event()
        self._shutdown = threading.Lock()
        self._takeover_lock = threading.Lock()
        self._takeover_pending = False
        self._experiment = None
        self._sessions = {}
        self._active, self._loaded = None, None
        self._recover_until = 0.0
        self._connections = OrderedDict()
        self._busy_connections = set()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.listen(max_clients)
        self.socket.settimeout(0.2)
        self.address = self.socket.getsockname()

    def serve_forever(self):
        log.info("QuICK TCP listening on %s:%d", *self.address)
        while not self._stop.is_set():
            try:
                conn, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            cancel_only = False
            if not self._slots.acquire(False):
                cancel_only = True
                if not self._cancel_slots.acquire(False):
                    conn.close()
                    continue
            configure_socket(conn)
            conn.settimeout(2 if cancel_only else 5)
            with self._state:
                self._connections[conn] = cancel_only
            threading.Thread(target=self._session, args=(conn, cancel_only), daemon=True).start()

    @staticmethod
    def _close_connection(conn):
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()

    def _normal_slot(self):
        if self._slots.acquire(False):
            return True
        with self._state:
            idle = next((sock for sock, control in self._connections.items()
                         if not control and sock not in self._busy_connections), None)
            if idle is not None:
                # Transfer the permit; the retired session must not release it.
                self._connections.pop(idle)
        if idle is None:
            return False
        self._close_connection(idle)
        return True

    def _promote_connection(self, conn):
        if not self._normal_slot():
            return False
        with self._state:
            self._connections[conn] = False
        self._cancel_slots.release()
        conn.settimeout(5)
        return True

    def shutdown(self):
        if not self._shutdown.acquire(False):
            return
        self._stop.set()
        self.socket.close()
        with self._state:
            if self._active:
                self._active[1].set()
            connections = list(self._connections)
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        # Do not free DMA buffers while a cancelled acquisition still owns them.
        if self._board.acquire(timeout=10):
            try:
                try:
                    self.backend.start_src("internal")
                    self.backend.stop_tproc()
                    self.backend.reset_gens()
                finally:
                    close = getattr(self.backend, "close", None)
                    if close is not None:
                        close()
            except Exception:
                log.exception("Hardware shutdown failed")
            finally:
                self._board.release()
        else:
            log.error("Hardware still busy after shutdown; DMA buffers were not freed")

    def _session(self, conn, cancel_only=False):
        session = uuid.uuid4().hex
        with self._state:
            self._sessions[session] = conn
        try:
            while not self._stop.is_set():
                if not cancel_only and not select.select([conn], [], [], 0.2)[0]:
                    continue
                with self._state:
                    if conn not in self._connections:
                        return
                    self._busy_connections.add(conn)
                try:
                    kind, request_id, meta, arrays = recv_frame(conn)
                    if kind != REQUEST:
                        raise ProtocolError("Expected a request frame")
                    try:
                        takeover = meta.get("command") == "config" and meta.get("takeover") is True
                        if cancel_only and meta.get("command") != "stop" and not takeover:
                            if not self._promote_connection(conn):
                                raise BusyError("Normal sessions are busy; reserved connection permits stop or takeover")
                            cancel_only = False
                        self._dispatch(conn, session, request_id, meta, arrays)
                    except (ValueError, RuntimeError, TimeoutError, KeyError, OverflowError, TypeError, AttributeError, IndexError) as exc:
                        send_frame(conn, ERROR, request_id,
                                   {"error": type(exc).__name__, "message": str(exc)[:4096]})
                finally:
                    with self._state:
                        self._busy_connections.discard(conn)
                        cancel_only = self._connections.get(conn, cancel_only)
                if cancel_only:
                    return
        except (OSError, EOFError, ProtocolError):
            pass
        except Exception:
            with self._state:
                retired = conn not in self._connections
            if not retired:
                log.exception("TCP session failed")
        finally:
            with self._state:
                if self._active and self._active[0] == session:
                    self._active[1].set()
                slot = self._connections.pop(conn, None)
                self._busy_connections.discard(conn)
                self._sessions.pop(session, None)
                if self._experiment is not None and self._experiment[0] == session:
                    self._experiment = None
            conn.close()
            if slot is not None:
                (self._cancel_slots if slot else self._slots).release()

    def _dispatch(self, conn, session, request_id, meta, arrays):
        command = meta.get("command")
        if arrays and command != "upload":
            raise ProtocolError("This command has no array payload")
        if command == "config":
            takeover = meta.get("takeover", False)
            if type(takeover) is not bool:
                raise ValueError("takeover must be a boolean")
            if takeover:
                self._takeover(conn, session)
            send_frame(conn, RESPONSE, request_id, dict(cfg=self.cfg, firmware_id=self.firmware_id,
                       session_id=session, protocol=1, max_payload=MAX_PAYLOAD, takeover_complete=takeover))
        elif command == "upload":
            program_id, cached = self._upload(meta.get("plan"), arrays)
            send_frame(conn, RESPONSE, request_id, dict(program_id=program_id, cached=cached))
        elif command == "run":
            self._stream_run(conn, session, request_id, meta)
        elif command == "stop":
            with self._state:
                active = self._active
                stopped = bool(active and active[0] == meta.get("session_id"))
                if stopped:
                    active[1].set()
            send_frame(conn, RESPONSE, request_id, dict(stopped=stopped))
        elif command == "status":
            with self._state:
                recovery = max(0.0, self._recover_until-time.monotonic())
                status = dict(busy=self._active is not None or recovery > 0 or self._takeover_pending,
                              takeover_pending=self._takeover_pending, recovery_seconds=recovery, cached_programs=len(self.cache),
                              cache_bytes=self.cache_size, firmware_id=self.firmware_id)
            send_frame(conn, RESPONSE, request_id, status)
        elif command in ("read_mem", "reset_gens"):
            self._claim_board()
            try:
                if command == "read_mem":
                    memory = meta.get("mem_sel", "dmem")
                    if memory not in ("pmem", "wmem", "dmem"):
                        raise ValueError("Invalid memory selector")
                    capacity = self.cfg["tprocs"][0][memory + "_size"]
                    addr = _integer(meta.get("addr", 0), 0, capacity-1, "memory address")
                    length = _integer(meta.get("length"), 1, capacity-addr, "memory length")
                    data = self.backend.read_mem(length=length, mem_sel=memory, addr=addr)
                    send_frame(conn, RESPONSE, request_id, {}, {"data": data})
                else:
                    self.backend.reset_gens()
                    self._loaded = None
                    send_frame(conn, RESPONSE, request_id, {"reset": True})
            finally:
                self._board.release()
        else:
            raise ProtocolError("Unknown command")

    def _claim_board(self):
        if self._stop.is_set():
            raise CancelledError("Server is shutting down")
        if not self._board.acquire(False):
            raise BusyError("Board is acquiring")
        if self._stop.is_set():
            self._board.release()
            raise CancelledError("Server is shutting down")
        with self._state:
            recovery = self._recover_until-time.monotonic()
            takeover = self._takeover_pending
        if takeover:
            self._board.release()
            raise BusyError("A new experimental session is taking over")
        if recovery > 0:
            self._board.release()
            raise BusyError("ADC window recovering (%.6f seconds remaining)" % recovery)

    def _takeover(self, conn, session):
        deadline = time.monotonic()+30
        if not self._takeover_lock.acquire(timeout=30):
            raise BusyError("Previous session takeover did not finish")
        claimed = False
        try:
            with self._state:
                self._takeover_pending = True
                previous = {self._experiment[1]} if self._experiment is not None else set()
                self._experiment = None
                if self._active is not None:
                    self._active[1].set()
                    active_conn = self._sessions.get(self._active[0])
                    if active_conn is not None:
                        previous.add(active_conn)
            for old in previous-{conn}:
                self._close_connection(old)
            while True:
                if self._stop.is_set():
                    raise CancelledError("Server is shutting down")
                if time.monotonic() >= deadline:
                    raise BusyError("Board cleanup or ADC recovery exceeded the takeover deadline")
                if select.select([conn], [], [], 0)[0]:
                    if not conn.recv(1, socket.MSG_PEEK):
                        raise EOFError("New experimental session disconnected")
                    raise ProtocolError("Commands cannot overlap a session takeover")
                with self._state:
                    if self._active is not None:
                        self._active[1].set()
                    recovered = time.monotonic() >= self._recover_until
                if recovered and self._board.acquire(False):
                    with self._state:
                        recovered = time.monotonic() >= self._recover_until
                    if recovered:
                        claimed = True
                        break
                    self._board.release()
                time.sleep(0.01)
            self._loaded = None
            self.backend.start_src("internal")
            self.backend.stop_tproc()
            for ch in range(len(self.cfg["readouts"])):
                self.backend.enable_buf(ch, enable_avg=False, enable_buf=False)
            self.backend.reset_gens()
            with self._state:
                reserved = self._connections.get(conn) is True
            if reserved:
                while not self._promote_connection(conn):
                    if time.monotonic() >= deadline:
                        raise BusyError("No normal session slot became available for takeover")
                    time.sleep(0.01)
            with self._state:
                self._experiment = (session, conn)
        finally:
            if claimed:
                self._board.release()
            with self._state:
                self._takeover_pending = False
            self._takeover_lock.release()

    def _recovery_delay(self, readouts):
        delay = 0.0
        for ro in readouts:
            cfg = self.cfg["readouts"][ro["ch"]]
            clocks = []
            for name, fallback in (("f_output", 1.0), ("f_fabric", cfg.get("f_output", 1.0))):
                value = cfg.get(name, fallback)
                clocks.append(value if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else 1.0)
            delay = max(delay, (ro["length"]/clocks[0] + 4/clocks[1])*1e-6)
        return delay

    def _upload(self, plan, arrays):
        if not isinstance(plan, dict) or plan.get("schema") != 1:
            raise ValueError("Expected Mercator plan schema 1")
        if plan.get("firmware_id") != self.firmware_id:
            raise ValueError("Program was compiled for a different board configuration")
        plan = json.loads(json_bytes(plan))
        self._validate_plan(plan, arrays)
        digest = hashlib.sha256(json_bytes(plan))
        size = 0
        for name in sorted(arrays):
            arr = np.ascontiguousarray(arrays[name])
            digest.update(json_bytes([name, arr.dtype.str, list(arr.shape)]))
            digest.update(memoryview(arr).cast("B"))
            size += arr.nbytes
        program_id = digest.hexdigest()
        if size > self.cache_bytes:
            raise ValueError("Program exceeds cache capacity")
        with self._state:
            if program_id in self.cache:
                self.cache.move_to_end(program_id)
                return program_id, True
            while self.cache and (self.cache_size + size > self.cache_bytes or len(self.cache) >= 8):
                _, old = self.cache.popitem(last=False)
                self.cache_size -= old[2]
            self.cache[program_id] = (plan, arrays, size)
            self.cache_size += size
        return program_id, False

    def _validate_plan(self, plan, arrays):
        if set(plan) - {"schema", "firmware_id", "loop_dims", "counter_addr", "generators", "readouts", "pfbs", "envelopes", "weights"}:
            raise ValueError("Unknown plan fields")
        loops = plan.get("loop_dims")
        if not isinstance(loops, list) or not 1 <= len(loops) <= 16:
            raise ValueError("Invalid loop dimensions")
        for value in loops:
            _integer(value, 1, 2**31-1, "loop dimension")
        _integer(math.prod(loops), 1, 2**31-1, "total shots")
        _integer(plan.get("counter_addr"), 1, 2, "counter register")
        expected = set()
        for memory in ("pmem", "wmem", "dmem"):
            if memory not in arrays:
                if memory == "pmem":
                    raise ValueError("Program memory is required")
                continue
            arr = arrays[memory]
            if arr.dtype != np.dtype("<i4") or arr.ndim != (1 if memory == "dmem" else 2):
                raise ValueError("Invalid %s format" % memory)
            if memory != "dmem" and arr.shape[1] != 8:
                raise ValueError("Memory rows must contain eight words")
            if not 1 <= len(arr) <= self.cfg["tprocs"][0][memory + "_size"]:
                raise ValueError("Memory capacity exceeded")
            expected.add(memory)
        for name, cfgkey in (("generators", "gens"), ("readouts", "readouts")):
            values = plan.get(name, [])
            if not isinstance(values, list) or len(values) > len(self.cfg[cfgkey]):
                raise ValueError("Invalid channel declarations")
            channels = set()
            for item in values:
                if not isinstance(item, dict):
                    raise ValueError("Channel declarations must be objects")
                ch = _integer(item.get("ch"), 0, len(self.cfg[cfgkey])-1, "channel")
                if ch in channels:
                    raise ValueError("Duplicate channel declaration")
                channels.add(ch)
                if name == "generators":
                    item.setdefault("nqz", 1)
                    _integer(item["nqz"], 1, 2, "Nyquist zone")
                    if "mixer_freq" in item and (not isinstance(item["mixer_freq"], (int, float)) or not math.isfinite(item["mixer_freq"])):
                        raise ValueError("Invalid mixer frequency")
                    if "mux_tones" in item and (not isinstance(item["mux_tones"], list) or
                            len(item["mux_tones"]) > self.cfg["gens"][ch].get("n_tones", 0) or
                            any(not isinstance(tone, dict) or "freq_int" not in tone for tone in item["mux_tones"])):
                        raise ValueError("Invalid mux tones")
                else:
                    _integer(item.get("length"), 1, 2**31-1, "readout length")
                    _integer(item.get("trigs"), 0, 2**31-1, "readout triggers")
                    cfg = self.cfg["readouts"][ch]
                    if "tproc_ctrl" not in cfg and "pfb_port" not in cfg and not isinstance(item.get("ro_config"), dict):
                        raise ValueError("Static readout configuration is required")
        paths = {r.get("ro_fullpath") for r in self.cfg["readouts"] if "pfb_port" in r}
        if not isinstance(plan.get("pfbs", []), list):
            raise ValueError("PFB declarations must be a list")
        for pfb in plan.get("pfbs", []):
            if not isinstance(pfb, dict):
                raise ValueError("PFB declarations must be objects")
            if pfb.get("path") not in paths or not isinstance(pfb.get("regs"), list):
                raise ValueError("Invalid PFB readout")
        for name, cfgkey, capacity in (("envelopes", "gens", "maxlen"), ("weights", "readouts", "wgt_maxlen")):
            if not isinstance(plan.get(name, []), list):
                raise ValueError("Envelope/weight declarations must be lists")
            for item in plan.get(name, []):
                if not isinstance(item, dict):
                    raise ValueError("Envelope/weight declarations must be objects")
                ch = _integer(item.get("ch"), 0, len(self.cfg[cfgkey])-1, "channel")
                cfg = self.cfg[cfgkey][ch]
                key = item.get("array")
                if key not in arrays or key in expected:
                    raise ValueError("Missing or duplicate envelope/weight array")
                arr = arrays[key]
                if arr.dtype != np.dtype("<i2") or arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError("Envelope/weight array must be int16 IQ")
                addr = _integer(item.get("addr", 0), 0, cfg.get(capacity, 0), "envelope address")
                item.setdefault("addr", 0)
                if len(arr) < 1 or addr + len(arr) > cfg.get(capacity, 0):
                    raise ValueError("Envelope/weight memory capacity exceeded")
                if name == "envelopes" and len(arr) % cfg["samps_per_clk"]:
                    raise ValueError("Envelope length is not a whole fabric cycle")
                expected.add(key)
        if set(arrays) != expected:
            raise ValueError("Unexpected program arrays")

    def _stream_run(self, conn, session, request_id, meta):
        mode = meta.get("mode")
        if mode not in ("accumulated", "decimated", "run"):
            raise ValueError("Invalid acquisition mode")
        rounds = _integer(meta.get("rounds", 1), 1, 1000000, "round count")
        if meta.get("start_src", "internal") not in ("internal", "external"):
            raise ValueError("Invalid start source")
        timeout = meta.get("timeout")
        if timeout is not None and (not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Timeout must be positive seconds")
        if type(meta.get("load_envelopes", True)) is not bool:
            raise ValueError("load_envelopes must be a boolean")
        with self._state:
            entry = self.cache.get(meta.get("program_id"))
        if entry is None:
            raise CacheMissError("Program is not cached; upload it again")
        plan, arrays, _ = entry
        total, readouts = math.prod(plan["loop_dims"]), plan.get("readouts", [])
        if mode != "run" and (not readouts or any(r["trigs"] <= 0 for r in readouts)):
            raise ValueError("Acquisition needs triggered readouts")
        for ro in readouts:
            cfg = self.cfg["readouts"][ro["ch"]]
            if mode == "accumulated" and ro["trigs"] > cfg["avg_maxlen"]-2:
                raise ValueError("One shot exceeds the accumulated buffer")
            if mode == "decimated" and (ro["length"] >= cfg["buf_maxlen"] or
                    total*ro["trigs"]*ro["length"] > cfg["buf_maxlen"] or total*ro["trigs"] > cfg["avg_maxlen"]-2):
                raise ValueError("Decimated acquisition exceeds a capture buffer")
        self._claim_board()
        cancel, done, output = threading.Event(), threading.Event(), queue.Queue(self.queue_depth)
        state = {"error": None}
        with self._state:
            self._active = (session, cancel)
            if self._stop.is_set():
                cancel.set()
        worker = threading.Thread(target=self._capture,
                                  args=(meta, entry, cancel, done, output, state), daemon=True)
        try:
            worker.start()
        except BaseException:
            with self._state:
                self._active = None
            self._board.release()
            raise
        try:
            last_sent = time.monotonic()
            poll_interval = min(0.05, self.heartbeat_interval)
            while not done.is_set() or not output.empty():
                if state["error"] is not None:
                    # Report errors only after the worker has disarmed hardware.
                    done.wait(poll_interval)
                    if done.is_set():
                        raise state["error"]
                    if time.monotonic()-last_sent >= self.heartbeat_interval:
                        send_frame(conn, DATA, request_id, {"kind": "heartbeat"})
                        last_sent = time.monotonic()
                    continue
                try:
                    kind, info, data = output.get(timeout=poll_interval)
                except queue.Empty:
                    if select.select([conn], [], [], 0)[0]:
                        if not conn.recv(1, socket.MSG_PEEK):
                            raise EOFError("Acquisition client disconnected")
                        raise ProtocolError("Commands cannot overlap an acquisition stream")
                    if time.monotonic()-last_sent >= self.heartbeat_interval:
                        send_frame(conn, DATA, request_id, {"kind": "heartbeat"})
                        last_sent = time.monotonic()
                    continue
                send_frame(conn, kind, request_id, info, data)
                last_sent = time.monotonic()
            if state["error"] is not None:
                raise state["error"]
            # The worker releases its board lease before signalling completion.
            send_frame(conn, DONE, request_id, dict(event="run", rounds=rounds))
        finally:
            cancel.set()
            worker.join(timeout=2)

    def _capture(self, meta, entry, cancel, done, output, state):
        plan, arrays, _ = entry
        rounds, mode = meta.get("rounds", 1), meta["mode"]
        total, readouts = math.prod(plan["loop_dims"]), plan.get("readouts", [])
        prepared = False

        def emit(kind, info, data=None):
            self._check(cancel, None)
            try:
                output.put((kind, info, data or {}), timeout=0.01)
            except queue.Full:
                raise OverflowError("TCP output queue filled; acquisition stopped without dropping data")

        try:
            self._check(cancel, None)
            self._configure(meta["program_id"], plan, arrays, meta.get("load_envelopes", True))
            for index in range(rounds):
                self._check(cancel, None)
                for ro in readouts:
                    if mode != "run":
                        self.backend.config_avg(ro["ch"], length=ro["length"],
                            edge_counting=ro.get("edge_counting", False),
                            high_threshold=ro.get("high_threshold", 0), low_threshold=ro.get("low_threshold", 0))
                        if mode == "decimated":
                            self.backend.config_buf(ro["ch"], length=ro["length"])
                        self.backend.enable_buf(ro["ch"], enable_avg=True, enable_buf=mode == "decimated")
                self.backend.reload_mem()
                self.backend.clear_tproc_counter(addr=plan["counter_addr"])
                prepared = True
                self.backend.prepare_round()
                self.backend.start_src(meta.get("start_src", "internal"))
                deadline = None if meta.get("timeout") is None else time.monotonic()+meta["timeout"]
                self.backend.start_tproc()
                if mode == "accumulated":
                    self._accumulated(plan, index, total, cancel, deadline, emit)
                else:
                    self._wait_counter(plan, total, cancel, deadline)
                    self.backend.start_src("internal")
                    if mode == "decimated":
                        for ro in readouts:
                            ch = ro["ch"]
                            n = total*ro["trigs"]*ro["length"]
                            data = self.backend.get_decimated(ch=ch, address=0, length=n)
                            emit(DATA, dict(kind="decimated", round=index, offset=0, count=n, channels=[ch]), {str(ch): data})
                        raw = {str(ro["ch"]): self.backend.get_accumulated(ch=ro["ch"], address=0, length=total*ro["trigs"]) for ro in readouts}
                        emit(DATA, dict(kind="raw", round=index, offset=0, count=total,
                                        channels=[r["ch"] for r in readouts]), raw)
                self.backend.start_src("internal")
                prepared = False
                self.backend.cleanup_round()
                emit(DONE, dict(event="round", round=index, shots=total))
        except BaseException as exc:
            state["error"] = exc if isinstance(exc, Exception) else RuntimeError("Acquisition interrupted")
            self._loaded = None
            # Disarming prevents new triggers, but avg_buffer finishes an
            # already active window. Keep its result out of the next run.
            for ro in readouts:
                try:
                    self.backend.enable_buf(ro["ch"], enable_avg=False, enable_buf=False)
                except Exception:
                    log.exception("Buffer cleanup failed")
            stopped = time.monotonic()
            with self._state:
                self._recover_until = max(self._recover_until, stopped+self._recovery_delay(readouts))
            for action in (lambda: self.backend.start_src("internal"), self.backend.stop_tproc,
                           self.backend.reset_gens):
                try:
                    action()
                except Exception:
                    log.exception("Acquisition cleanup failed")
        finally:
            if prepared:
                try:
                    self.backend.cleanup_round()
                except Exception:
                    log.exception("Round cleanup failed")
            with self._state:
                self._active = None
            self._board.release()
            done.set()

    def _configure(self, program_id, plan, arrays, load_envelopes):
        soc = self.backend
        soc.start_src("internal")
        soc.stop_tproc(lazy=True)
        if self._loaded == (program_id, True) or self._loaded == (program_id, load_envelopes):
            return
        if load_envelopes:
            for env in plan.get("envelopes", []):
                soc.load_envelope(env["ch"], data=arrays[env["array"]], addr=env["addr"])
            for weights in plan.get("weights", []):
                soc.load_weights(weights["ch"], data=arrays[weights["array"]], addr=weights.get("addr", 0))
        for gen in plan.get("generators", []):
            soc.set_nyquist(gen["ch"], gen["nqz"])
            if "mixer_freq" in gen:
                soc.set_mixer_freq(gen["ch"], gen["mixer_freq"])
            if "mux_tones" in gen:
                soc.config_mux_gen(gen["ch"], gen["mux_tones"])
        for ro in plan.get("readouts", []):
            cfg = self.cfg["readouts"][ro["ch"]]
            if "tproc_ctrl" not in cfg and "pfb_port" not in cfg:
                soc.configure_readout(ro["ch"], ro["ro_config"])
        for pfb in plan.get("pfbs", []):
            soc.config_mux_readout(pfb["path"], pfb["regs"], pfb.get("sel"))
        soc.load_bin_program({name: arrays.get(name) for name in ("pmem", "wmem", "dmem")}, load_mem=False)
        self._loaded = (program_id, load_envelopes)

    @staticmethod
    def _check(cancel, deadline):
        if cancel.is_set():
            raise CancelledError("Acquisition cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Acquisition timed out")

    def _wait_counter(self, plan, total, cancel, deadline):
        previous = 0
        while True:
            self._check(cancel, deadline)
            count = self.backend.get_tproc_counter(addr=plan["counter_addr"])
            if not previous <= count <= total:
                raise RuntimeError("Invalid or reset shot counter")
            if count == total:
                return
            previous = count
            time.sleep(0.0001)

    def _accumulated(self, plan, index, total, cancel, deadline, emit):
        readouts, offset = plan["readouts"], 0
        maxshots = min((self.cfg["readouts"][r["ch"]]["avg_maxlen"]-2)//r["trigs"] for r in readouts)
        stride = max(1, min(1024, maxshots//8))
        bytes_per_shot = sum(r["trigs"]*8 for r in readouts)
        stride = min(stride, max(1, MAX_PAYLOAD//bytes_per_shot))
        while offset < total:
            self._check(cancel, deadline)
            count = self.backend.get_tproc_counter(addr=plan["counter_addr"])
            if not offset <= count <= total:
                raise RuntimeError("Invalid or reset shot counter")
            available = count-offset
            if available > maxshots:
                raise OverflowError("Accumulated buffer overrun; increase the repetition delay")
            if available < stride and count != total:
                time.sleep(0.0001)
                continue
            n = min(available, stride)
            if not n:
                continue
            raw = {}
            for ro in readouts:
                ch, reads = ro["ch"], ro["trigs"]
                capacity = self.cfg["readouts"][ch]["avg_maxlen"]
                raw[str(ch)] = self.backend.get_accumulated(ch=ch, address=offset*reads % capacity, length=n*reads)
                if np.shape(raw[str(ch)]) != (n*reads, 2):
                    raise RuntimeError("Hardware returned an invalid IQ block")
            after = self.backend.get_tproc_counter(addr=plan["counter_addr"])
            if after-offset > maxshots:
                raise OverflowError("Accumulated buffer overwritten during DMA transfer")
            emit(DATA, dict(kind="accumulated", round=index, offset=offset, count=n,
                            channels=[r["ch"] for r in readouts]), raw)
            offset += n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON server and hardware configuration")
    parser.add_argument("--bitfile")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--attach", action="store_true", default=None, help="Attach to already loaded matching firmware")
    parser.add_argument("--dac-sample-rates", type=json.loads, help='JSON mapping of tile to MHz')
    parser.add_argument("--adc-sample-rates", type=json.loads, help='JSON mapping of tile to MHz')
    parser.add_argument("--queue-depth", type=int)
    args = parser.parse_args()
    settings = {}
    if args.config:
        with open(args.config) as handle:
            settings = json.load(handle)
        if not isinstance(settings, dict):
            parser.error("config must be a JSON object")
    for name in ("bitfile", "host", "port", "dac_sample_rates", "adc_sample_rates", "queue_depth"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if not settings.get("bitfile"):
        parser.error("--bitfile or config.bitfile is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .hardware import BoardHardware
    board = BoardHardware(settings["bitfile"], download=False if args.attach else settings.get("download", True),
                          dac_sample_rates=settings.get("dac_sample_rates"), adc_sample_rates=settings.get("adc_sample_rates"),
                          dma_timeout=settings.get("dma_timeout", 5), force_init_clks=settings.get("force_init_clks", False))
    server = QuickServer(board, host=settings.get("host", "0.0.0.0"), port=settings.get("port", 9999),
                         queue_depth=settings.get("queue_depth", 8))
    # Defer hardware/server locks until serve_forever exits its accept timeout.
    signal.signal(signal.SIGTERM, lambda signum, frame: server._stop.set())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
