"""Bounded JSON headers and contiguous numeric arrays over a byte stream."""

import json
import math
import socket
import struct
import numpy as np

REQUEST, RESPONSE, DATA, DONE, ERROR = range(1, 6)
HEADER = struct.Struct("!4sBBHIIQ")
MAGIC, VERSION = b"QK01", 1
MAX_METADATA, MAX_PAYLOAD, MAX_ARRAYS = 1024 * 1024, 64 * 1024 * 1024, 4096
DTYPES = {np.dtype(t).str for t in ("i1", "u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8")}


class ProtocolError(ValueError):
    pass


def configure_socket(sock):
    """Use low-latency TCP and bounded dead-peer detection where supported."""
    options = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
               (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (("TCP_KEEPIDLE", 10), ("TCP_KEEPINTVL", 2),
                        ("TCP_KEEPCNT", 3), ("TCP_USER_TIMEOUT", 10000)):
        option = getattr(socket, name, None)
        if option is not None:
            options.append((socket.IPPROTO_TCP, option, value))
    for level, option, value in options:
        try:
            sock.setsockopt(level, option, value)
        except OSError:
            pass
    if hasattr(socket, "SIO_KEEPALIVE_VALS") and hasattr(sock, "ioctl"):
        try:
            sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 10000, 2000))
        except OSError:
            pass
    return sock


def _json_scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError("Not JSON data: %s" % type(value).__name__)


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True,
                      allow_nan=False, default=_json_scalar).encode("utf-8")


def prepare_frame(kind, request_id, metadata=None, arrays=None):
    """Return header, JSON bytes, and zero-copy byte views of array payloads."""
    if kind not in (REQUEST, RESPONSE, DATA, DONE, ERROR):
        raise ProtocolError("Unknown frame kind")
    if not isinstance(request_id, int) or not 0 <= request_id < 2**32:
        raise ProtocolError("Invalid request ID")
    meta = {} if metadata is None else dict(metadata)
    if "_arrays" in meta:
        raise ProtocolError("_arrays is a reserved metadata key")
    arrays = {} if arrays is None else arrays
    if len(arrays) > MAX_ARRAYS:
        raise ProtocolError("Too many arrays")
    descriptors, payloads, size = [], [], 0
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ProtocolError("Invalid array name")
        arr = np.asarray(value)
        dtype = arr.dtype.newbyteorder("<")
        if dtype.str not in DTYPES or arr.ndim > 8:
            raise ProtocolError("Unsupported numeric array type or rank")
        arr = np.ascontiguousarray(arr, dtype=dtype)
        if size + arr.nbytes > MAX_PAYLOAD:
            raise ProtocolError("Frame payload exceeds limit")
        descriptors.append(dict(name=name, dtype=dtype.str, shape=list(arr.shape),
                                offset=size, nbytes=arr.nbytes))
        payloads.append(memoryview(arr).cast("B"))
        size += arr.nbytes
    meta["_arrays"] = descriptors
    encoded = json_bytes(meta)
    if len(encoded) > MAX_METADATA:
        raise ProtocolError("Frame metadata exceeds limit")
    return HEADER.pack(MAGIC, VERSION, kind, 0, request_id, len(encoded), size), encoded, payloads


def send_frame(sock, kind, request_id, metadata=None, arrays=None):
    header, encoded, payloads = prepare_frame(kind, request_id, metadata, arrays)
    sock.sendall(header + encoded)
    for payload in payloads:
        if payload:
            sock.sendall(payload)


def _receive(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if not received:
            raise EOFError("Peer closed the connection")
        offset += received
    return data


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate JSON key")
        result[key] = value
    return result


def recv_frame(sock):
    header = HEADER.unpack(_receive(sock, HEADER.size))
    magic, version, kind, flags, request_id, metadata_size, payload_size = header
    if magic != MAGIC or version != VERSION or flags or kind not in (REQUEST, RESPONSE, DATA, DONE, ERROR):
        raise ProtocolError("Unsupported frame header")
    if metadata_size > MAX_METADATA or payload_size > MAX_PAYLOAD:
        raise ProtocolError("Frame exceeds size limit")
    try:
        meta = json.loads(_receive(sock, metadata_size), object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError("Nonfinite JSON number")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("Invalid frame metadata") from exc
    if not isinstance(meta, dict):
        raise ProtocolError("Frame metadata must be an object")
    descriptors = meta.pop("_arrays", [])
    if not isinstance(descriptors, list) or len(descriptors) > MAX_ARRAYS:
        raise ProtocolError("Invalid array descriptors")
    checked, names, offset = [], set(), 0
    for desc in descriptors:
        if not isinstance(desc, dict) or set(desc) != {"name", "dtype", "shape", "offset", "nbytes"}:
            raise ProtocolError("Invalid array descriptor")
        name, dtype, shape = desc["name"], desc["dtype"], desc["shape"]
        if not isinstance(name, str) or not name or len(name) > 128 or name in names:
            raise ProtocolError("Invalid or duplicate array name")
        if not isinstance(dtype, str) or dtype not in DTYPES:
            raise ProtocolError("Unsupported array dtype")
        if not isinstance(shape, list) or len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
            raise ProtocolError("Invalid array shape")
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if type(desc["offset"]) is not int or type(desc["nbytes"]) is not int or desc["offset"] != offset or desc["nbytes"] != size:
            raise ProtocolError("Invalid array extent")
        offset += size
        if offset > payload_size:
            raise ProtocolError("Array extends beyond payload")
        checked.append((name, dtype, shape, desc["offset"], size))
        names.add(name)
    if offset != payload_size:
        raise ProtocolError("Undescribed payload bytes")
    payload = _receive(sock, payload_size)
    arrays = {name: np.ndarray(shape, dtype=dtype, buffer=payload, offset=start)
              for name, dtype, shape, start, size in checked}
    return kind, request_id, meta, arrays
