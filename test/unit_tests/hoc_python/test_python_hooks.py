# The hooks of nrn_python_hook_set (docs/capi.rst), with C functions of known results in place of
# the interface's, which are put back after each test.
import ctypes

import pytest
from neuron import h
from neuron._interface import _provider, api

lib = api.libnrniv
hook_set = lib.nrn_python_hook_set
hook_set.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
bytes_set = lib.nrn_bytes_set
bytes_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
call = lib.nrn_function_call_nothrow
call.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
symbol = lib.nrn_symbol
symbol.argtypes, symbol.restype = [ctypes.c_char_p], ctypes.c_void_p
data = lib.nrn_vector_data
data.argtypes, data.restype = [ctypes.c_void_p], ctypes.POINTER(ctypes.c_double)
push = lib.nrn_object_push
push.argtypes = [ctypes.c_void_p]
new_python_object = (
    api._new_python_object
)  # nrn_python_object_new, with a reference to give
ERR = (ctypes.c_void_p, ctypes.c_size_t)  # c_char_p would copy the buffer
PRAXIS = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.py_object,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    *ERR,
)


@pytest.fixture
def replace():
    replaced = []

    def set_hook(name, callback):
        replaced.append(name)
        assert (
            hook_set(name.encode(), ctypes.cast(callback, ctypes.c_void_p), None, 0)
            == 0
        )

    yield set_hook
    for name in replaced:
        hook_set(
            name.encode(), ctypes.cast(_provider._HOOKS[name], ctypes.c_void_p), None, 0
        )


def test_names():
    error = ctypes.create_string_buffer(64)
    assert hook_set(b"praxis", None, error, len(error)) == 0
    hook_set(
        b"praxis", ctypes.cast(_provider._HOOKS["praxis"], ctypes.c_void_p), None, 0
    )
    assert hook_set(b"nosuch", None, error, len(error)) == 1
    assert error.value == b"nosuch is not a Python hook"


def test_praxis_value(replace):
    seen = []

    @PRAXIS
    def praxis(payload, vec, x, err, size):
        v = data(vec)
        seen.append((payload, v[0], v[1]))
        x[0] = 42.5 + (v[0] - 1) ** 2 + (v[1] + 2) ** 2
        return 1

    replace("praxis", praxis)
    h.attr_praxis(1e-8, 0.5, 0)
    f, vec = (lambda v: 0), h.Vector(2)
    assert abs(h.fit_praxis(f, vec) - 42.5) < 1e-12
    assert abs(vec[0] - 1) < 1e-5 and abs(vec[1] + 2) < 1e-5
    assert seen[0] == (f, 0.0, 0.0) and all(s[0] is f for s in seen)


def test_failure_message(replace):
    @PRAXIS
    def praxis(payload, vec, x, err, size):
        ctypes.memmove(err, b"boom\0", 5)
        return 0

    replace("praxis", praxis)
    vec = h.Vector(2)
    f = new_python_object(lambda v: 0)
    push(f)
    push(vec._obj)
    error = ctypes.create_string_buffer(1024)
    assert call(symbol(b"fit_praxis"), 2, error, len(error)) == 1
    assert error.value == b"hoc_execerror: boom"
    lib.nrn_object_unref(ctypes.c_void_p(f))


def test_pickle_bytes(replace):
    received = []

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.py_object, ctypes.c_void_p, *ERR)
    def pickle(payload, out, err, size):
        bytes_set(out, b"\x01\x02\x00\x04\x05", 5)
        return 1

    @ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        *ERR,
    )
    def unpickle(data, size, result, err, errsize):
        received.append(ctypes.string_at(data, size))
        result[0] = new_python_object("unpickled")
        return 1

    replace("pickle", pickle)
    replace("unpickle", unpickle)
    pc = h.ParallelContext()
    pc.pack([1, 2])
    pc.post("python_hooks")
    pc.take("python_hooks")
    assert pc.upkpyobj() == "unpickled"
    assert received == [b"\x01\x02\x00\x04\x05"]
