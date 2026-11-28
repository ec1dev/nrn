"""What NEURON calls when HOC or the simulator uses Python (docs/capi.rst, "Python").

NEURON keeps one C function per hook, registered by name with
nrn_python_hook_set. These are ctypes callbacks. Each returns 1 when it is done
and 0 when it failed, with the message of the HOC error in ``err``; NEURON
raises that error after the callback has returned, so nothing is thrown through
a Python frame. 2 and 3 say more where a hook documents it. NEURON holds the
PythonObject of the call and puts HOC at its top level before it calls.

A Python exception that becomes a HOC error is kept for the Python code that
called HOC, which raises it when its call returns (take_exception), if HOC's
caller is a _nothrow function of the C API (nrn_hoc_error_is_returned) and
there is such Python code; otherwise it is printed here, as sys.excepthook
prints it, and SystemExit ends the process.
"""

import ctypes
import math
import operator
import os
import pickle
import sys
import traceback

from . import _coreneuron
from .api import (
    CALLS_PYTHON,
    libnrniv,
    EPOCH,
    HOC_LOCK,
    OBJECT_BASE,
    SECTION_BASE,
    SEGMENT_BASE,
    _PyNumber_Check,
    _PySequence_Check,
    _bind,
    _decode_hoc_string,
    _is_hoc_number,
    _new_python_object,
    _nrn_section_is_active,
    _nrn_symbol_name,
    _symbol_pop,
    _nrn_double_pop,
    _nrn_double_ptr_pop,
    _nrn_double_push,
    _nrn_int_pop,
    _nrn_object_pop,
    _nrn_object_push,
    _nrn_object_ref,
    _nrn_object_unref,
    _nrn_stack_type,
    _nrn_str_pop0,
    _nrn_str_push,
    _nrn_vector_data,
)
from .nrnref import reference
from .sections import Segment, _section_for
from .utils import _wrap_owned_object

c_int, c_double, c_size_t, c_void_p = (
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_size_t,
    ctypes.c_void_p,
)
py_object = ctypes.py_object
_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
c_char_p = ctypes.c_char_p

if CALLS_PYTHON:
    _hook_set = _bind(
        "nrn_python_hook_set", c_int, c_char_p, c_void_p, c_void_p, c_size_t
    )
    _bytes_set = _bind("nrn_bytes_set", None, c_void_p, c_void_p, c_size_t)
    _is_returned = _bind("nrn_hoc_error_is_returned", c_int)
    # Its own function object: api.py's has argtypes of its own.
    _payload_of = ctypes.PYFUNCTYPE(c_int, c_void_p, ctypes.POINTER(c_void_p))(
        ("nrn_python_object_get", libnrniv)
    )
_Py_IncRef = ctypes.PYFUNCTYPE(None, py_object)(("Py_IncRef", ctypes.pythonapi))
_Py_DecRef = ctypes.PYFUNCTYPE(None, c_void_p)(("Py_DecRef", ctypes.pythonapi))
_cast = ctypes.cast
try:  # the object at an address: a third of ctypes.cast(p, py_object).value
    _object_at = ctypes.PYFUNCTYPE(py_object, c_void_p)(("Py_NewRef", ctypes.pythonapi))
except AttributeError:  # Python 3.9

    def _object_at(p):
        return _cast(p, py_object).value


_PyMapping_Check = ctypes.PYFUNCTYPE(c_int, py_object)(
    ("PyMapping_Check", ctypes.pythonapi)
)

_FILE = os.path.normcase(os.path.abspath(__file__))

# Stack types (nrn_stack_types_t).
_STR, _VAR, _NUM, _OBJVAR, _OBJTMP, _INT = 1, 2, 3, 4, 5, 6


def _write(err, size, message):
    """The message of a failed hook, NUL-terminated, at most size bytes."""
    data = message.encode("utf-8", "replace")[: size - 1]
    ctypes.memmove(err, data + b"\0", len(data) + 1)


def _bytes(out, data):
    """The result of a hook: bytes, or a ctypes array."""
    _bytes_set(out, data, len(data) if data.__class__ is bytes else ctypes.sizeof(data))


# ---- errors

# What a failed call from Python raises when it has returned (take_exception).
_kept = None
# The code objects of the hooks: the frame below the innermost one is the Python
# code that called HOC, if any.
_ENTRIES = set()

EXCEPTION, HOC_ERROR, ALWAYS = range(3)


def _strip(exc):
    """The traceback from the first frame that is not this module's: the user's
    callable, as NEURON called it."""
    tb = exc.__traceback__
    while tb is not None and os.path.normcase(tb.tb_frame.f_code.co_filename) == _FILE:
        tb = tb.tb_next
    exc.__traceback__ = tb
    return exc


def _has_caller():
    frame = sys._getframe(1)
    while frame is not None and frame.f_code not in _ENTRIES:
        frame = frame.f_back
    return frame is not None and frame.f_back is not None


def print_exception(exc):
    """As PyErr_Print: sys.excepthook, and SystemExit ends the process."""
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is None:
            status = 0
        elif isinstance(code, int):
            status = code
        else:
            try:
                print(code, file=sys.stderr)
            except Exception:
                pass
            status = 1
        ctypes.pythonapi.Py_Exit(status)
    sys.last_exc, sys.last_type, sys.last_value = exc, type(exc), exc
    sys.last_traceback = exc.__traceback__
    try:
        sys.excepthook(type(exc), exc, exc.__traceback__)
    except BaseException as hook_error:
        print("Error in sys.excepthook:", file=sys.stderr)
        sys.__excepthook__(type(hook_error), hook_error, hook_error.__traceback__)
        print("\nOriginal exception was:", file=sys.stderr)
        sys.__excepthook__(type(exc), exc, exc.__traceback__)


def _formatted(exc):
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _keep(exc):
    global _kept
    old, _kept = _kept, exc
    if old is not None:
        print_exception(old)


def take_exception():
    """The exception a failed call from Python is to raise, or None."""
    global _kept
    exc, _kept = _kept, None
    return exc


def _fail(exc, message, what, err, size, show=True):
    """Make a Python error the HOC error `message`. If no Python code can get the
    exception it is printed (unless not show), and SystemExit ends the process.
    The Python code that called HOC gets the exception (EXCEPTION), or the HOC
    error as NEURON's Python module raised it, a RuntimeError of the message
    (HOC_ERROR, but not for SystemExit; ALWAYS)."""
    exit_ = isinstance(exc, SystemExit)
    if _is_returned() and _has_caller():
        if what == ALWAYS or (what == HOC_ERROR and not exit_):
            if show:
                _stderr(_formatted(exc))
            exc = RuntimeError("hocobj_call error: hoc_execerror: " + message)
        _keep(exc)
    elif show or (exit_ and what != ALWAYS):
        print_exception(exc)
    _write(err, size, message)
    return 0


def _stderr(text):
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass


def _fail_hook(exc, err, size):
    """A hook's own failure: the text is enough."""
    message = str(exc) or type(exc).__name__
    return _fail(exc, message, HOC_ERROR, err, size, False)


def _fail_callback(exc, message, err, size, with_traceback=False):
    """A callback failed. KeyboardInterrupt and the like do not stop NEURON:
    they are printed, and the hook returns 1."""
    if not isinstance(exc, (Exception, SystemExit)):
        print_exception(exc)
        return 1
    if with_traceback:
        return _fail(exc, message + "\n" + _formatted(exc), HOC_ERROR, err, size, False)
    return _fail(exc, message, HOC_ERROR, err, size)


def _error_printed(exc, message, err, size):
    """PyErr_Print, then the HOC error, as NEURON's Python module did."""
    print_exception(exc)
    _write(err, size, message)
    return 0


def _error_traceback(exc, message, err, size):
    """A failed call, as NEURON's Python module reported it: an Exception with its
    traceback, anything else as PyErr_Print does; then the HOC error."""
    if isinstance(exc, Exception):
        _stderr(_formatted(exc) + "\n")
    else:
        print_exception(exc)
    _write(err, size, message)
    return 0


# ---- values


def _payload(p):
    """What a PythonObject holds, from its this_pointer: __main__ for NULL."""
    return sys.modules["__main__"] if not p else _object_at(p)


def _to_python(obj):
    """The Python value of a borrowed HOC Object*."""
    if not obj:
        return None
    _nrn_object_ref(obj)
    return _wrap_owned_object(obj)


def _wrapped(value):
    """The Object* of a wrapper, or 0 for one whose construction failed."""
    try:
        return _OBJ(value) or 0
    except AttributeError:
        return 0


_OBJ = OBJECT_BASE._obj.__get__


def to_hoc(value):
    """The HOC object for a Python value, with a reference for the caller; 0 for None."""
    if value is None:
        return 0
    if isinstance(value, OBJECT_BASE):
        obj = _wrapped(value)
        if obj:
            _nrn_object_ref(obj)
        return obj
    return _new_python_object(value)


def _pop():
    """Pop a value from the HOC stack, as Python."""
    kind = _nrn_stack_type()
    if kind == _NUM:
        return _nrn_double_pop()
    if kind == _STR:
        return _decode_hoc_string(_nrn_str_pop0())
    if kind == _VAR:
        pointer = _nrn_double_ptr_pop()
        if not pointer:
            raise _HocError("Invalid pointer argument")
        return pointer[0]
    if kind == _OBJVAR or kind == _OBJTMP:
        obj = _nrn_object_pop()
        return _wrap_owned_object(obj) if obj else None
    raise _HocError("This type of argument can not be passed to Python")


class _HocError(Exception):
    """A plain HOC error: its message, nothing printed."""


def _pop_arguments(n):
    """Pop n arguments, the last one first."""
    args = [None] * n
    while n:
        n -= 1
        args[n] = _pop()
    return args


def _discard_pops(n):
    """Pop what is left after a pop failed, keeping the stack balanced."""
    for _ in range(n):
        try:
            if _nrn_stack_type() in (_OBJVAR, _OBJTMP):
                obj = _nrn_object_pop()
                if obj:
                    _nrn_object_unref(obj)
            elif _nrn_stack_type() == _STR:
                _nrn_str_pop0()
            elif _nrn_stack_type() == _VAR:
                _nrn_double_ptr_pop()
            else:
                _nrn_double_pop()
        except BaseException:
            return


# HOC strings pushed for a component: they outlive the expression that uses them,
# as hoc_temp_charptr's do.
_STRINGS = [None] * 128
_next_string = [0]


def _temp_string(value):
    """A char** for a str or bytes, up to its first NUL; None for anything else."""
    if isinstance(value, str):
        data = value.encode("utf-8")
    elif isinstance(value, bytes):
        data = value
    else:
        return None
    pointer = ctypes.c_char_p(data.split(b"\0", 1)[0])
    i = _next_string[0] = (_next_string[0] + 1) % len(_STRINGS)
    _STRINGS[i] = pointer
    return ctypes.pointer(pointer)


def _push(value):
    """Push a component's value, in place of the object HOC looked inside."""
    if value.__class__ is float or _is_hoc_number(value):
        x = float(value)
        _discard_looked_at()
        _nrn_double_push(x)
        return
    string = _temp_string(value)
    if string is not None:
        _discard_looked_at()
        _nrn_str_push(string)
        return
    obj = to_hoc(value)
    _discard_looked_at()
    _nrn_object_push(obj or None)
    if obj:
        _nrn_object_unref(obj)


def _discard_looked_at():
    obj = _nrn_object_pop()
    if obj:
        _nrn_object_unref(obj)


# ---- the hooks


def _hook(name, restype, *argtypes):
    def register(function):
        _ENTRIES.add(function.__code__)
        callback = ctypes.CFUNCTYPE(restype, *argtypes)(function)
        _HOOKS[name] = callback
        return function

    return register


# name -> the ctypes callback, kept for the life of the process: NEURON has only
# its address.
_HOOKS = {}
ERR = (c_void_p, c_size_t)

# Names as Python keeps them, per symbol name.
_names = {}


def _attribute(payload, name, top):
    """What HOC writes as o.name. `new PythonObject()` stands for __main__, where
    builtins are found too."""
    if top:
        main = sys.modules["__main__"]
        variables = main.__dict__
        if name in variables:
            return variables[name]
        builtins = variables.get("__builtins__")
        builtins = getattr(builtins, "__dict__", builtins)
        if isinstance(builtins, dict) and name in builtins:
            return builtins[name]
        return getattr(main, name)
    if name == "_":
        return payload
    return getattr(payload, name)


def _check_dimension(n):
    if n > 1:
        raise _HocError(
            f"{n} dimensional python objects can't be accessed from hoc with "
            "var._[i1][i2]... syntax. Must use var._[i1]._[i2]... hoc syntax."
        )


def _key():
    """An index from the HOC stack: a number is truncated, as HOC's are."""
    if _nrn_stack_type() == _NUM:
        x = _nrn_double_pop()
        return int(x) if math.isfinite(x) else x
    return _pop()


@_hook("component", c_int, c_void_p, c_void_p, c_char_p, c_int, c_int, *ERR)
def component(obj, held, sym_name, nindex, isfunc, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    name = _names.get(sym_name)
    if name is None:
        name = _names[sym_name] = sys.intern(sym_name.decode("utf-8"))
    payload = _payload(held)
    try:
        value = _attribute(payload, name, not held)
    except BaseException as exc:
        return _error_printed(_strip(exc), "No attribute: " + name, err, size)
    if isfunc:
        try:
            args = _pop_arguments(nindex)
        except _HocError as exc:
            _write(err, size, str(exc))
            return 0
        except BaseException as exc:
            return _fail(_strip(exc), "not a Python string", EXCEPTION, err, size)
        try:
            value = value(*args)
        except BaseException as exc:
            _strip(exc)
            if not isinstance(exc, Exception):
                # As NEURON's Python module: printed, and nothing is pushed, which
                # HOC then reports as its own error.
                print_exception(exc)
                return 1
            return _error_traceback(
                exc, "PyObject method call failed: " + name, err, size
            )
    elif nindex:
        try:
            _check_dimension(_nrn_int_pop())
            key = _key()
        except _HocError as exc:
            _write(err, size, str(exc))
            return 0
        except BaseException as exc:
            return _fail(_strip(exc), "not a Python string", EXCEPTION, err, size)
        try:
            value = value[key]
        except BaseException as exc:
            # 3: NEURON adds the object's name to the message.
            _error_printed(_strip(exc), "Python get item failed:", err, size)
            return 3
    try:
        _push(value)
    except BaseException as exc:
        return _fail(_strip(exc), "not a HOC string", EXCEPTION, err, size)
    return 1


@_hook("component_assign", c_int, c_void_p, c_void_p, *ERR)
def component_assign(obj, held, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        value = _pop()
    except _HocError as exc:
        _write(err, size, str(exc))
        return 0
    except BaseException as exc:
        return _fail(_strip(exc), "not a Python string", EXCEPTION, err, size)
    target = _nrn_object_pop()  # NEURON holds it for the call
    if target:
        _nrn_object_unref(target)
    name = _nrn_symbol_name(_symbol_pop()).decode("utf-8")
    nindex = _nrn_int_pop()
    try:
        _check_dimension(nindex)
        if nindex:
            _check_dimension(_nrn_int_pop())
            key = _key()
    except _HocError as exc:
        _write(err, size, str(exc))
        return 0
    except BaseException as exc:
        return _fail(_strip(exc), "not a Python string", EXCEPTION, err, size)
    payload = _payload(held)
    try:
        if nindex:
            _attribute(payload, name, not held)[key] = value
        else:
            setattr(payload, name, value)
    except BaseException as exc:
        return _error_printed(
            _strip(exc), "Assignment to PythonObject failed", err, size
        )
    return 1


@_hook("to_number", c_int, c_void_p, ctypes.POINTER(c_double))
def to_number(obj, x):
    """A HOC object as a number, for the GUI: 0 if it is none."""
    try:
        x[0] = float(_to_python(obj))
    except BaseException:
        return 0
    return 1


@_hook("release", None, c_void_p)
def release(held):
    """A PythonObject goes: the reference _new_python_object gave it."""
    EPOCH[0] = None  # a __del__ may run
    _Py_DecRef(held)


@_hook("hold", c_int, c_void_p, ctypes.POINTER(c_void_p))
def hold(held, result):
    """A new reference to what a PythonObject holds, for NEURON to keep (PlotShape)."""
    value = _payload(held)
    _Py_IncRef(value)
    result[0] = id(value)
    return 1


@_hook("operator", c_int, c_void_p, c_char_p, c_void_p, c_int, c_double, *ERR)
def operator_(payload, method, other, is_number, x, err, size):
    """HOC's operators try a Python object's method first: 1 when it pushed the
    result, 2 to leave the operator to HOC."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    if is_number:
        # A whole number is an int: rxd's coefficients tell them apart.
        arg = int(x) if x == math.floor(x) and abs(x) < 9e18 else x
    else:
        try:
            arg = _to_python(other)
        except BaseException as exc:
            return _fail_hook(_strip(exc), err, size)
    try:
        f = getattr(payload, method.decode())
        if not callable(f):
            return 2
        result = f(arg)
        number = _PyNumber_Check(result)
        if number:
            value = float(result)
    except Exception:
        return 2
    if number:
        _nrn_double_push(value)
        return 1
    obj = to_hoc(result)
    _nrn_object_push(obj or None)
    if obj:
        _nrn_object_unref(obj)
    return 1


@_hook("statement", c_int, c_char_p, ctypes.POINTER(c_int), *ERR)
def statement(text, ok, err, size):
    """nrnpython(): run statements in __main__; 0 only for SystemExit."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        code = compile(text, "<string>", "exec")
        exec(code, sys.modules["__main__"].__dict__)
    except BaseException as exc:
        _strip(exc)
        if isinstance(exc, SystemExit):
            return _fail(exc, "SystemExit", EXCEPTION, err, size)
        print_exception(exc)
        return 1
    ok[0] = 1
    return 1


def _call_command(command):
    """A callable, or (callable, arguments), or (callable, argument)."""
    if isinstance(command, tuple):
        args = command[1]
        return command[0](*args) if isinstance(args, tuple) else command[0](args)
    return command()


@_hook("command", c_int, c_void_p, *ERR)
def command(held, err, size):
    """hoccommand_exec, per event (NetCon.record, CVode.event, ...): 2 when it
    failed and that was only printed."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        _call_command(_payload(held))
    except BaseException as exc:
        message = "Python Callback failed [hoccommand_exec]:"
        return 2 if _fail_callback(_strip(exc), message, err, size, True) else 0
    return 1


@_hook("apply", c_int, c_void_p, c_void_p, c_int, c_int, *ERR)
def apply(held, vec, start, end, err, size):
    """Vector.apply: each element from start to end becomes f(element). As
    NEURON's nrnpy_call_func, any exception of the call or of the conversion,
    an interrupt and SystemExit too, is only the HOC error."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    f = _payload(held)
    if not callable(f):
        _write(err, size, "Object is not callable")
        return 0
    for i in range(start, end + 1):
        try:
            result = f(_nrn_vector_data(vec)[i])
        except BaseException as exc:
            message = "Python function call raised exception"
            return _fail(_strip(exc), message, ALWAYS, err, size, False)
        if not _PyNumber_Check(result):
            _write(err, size, "Expected a numeric result from Python function")
            return 0
        try:
            value = float(result)
        except BaseException as exc:
            message = "Failed to convert result to float"
            return _fail(_strip(exc), message, ALWAYS, err, size, False)
        _nrn_vector_data(vec)[i] = value  # f may have resized it
    return 1


@_hook("call_string", c_int, c_void_p, c_void_p, *ERR)
def call_string(payload, out, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        result = _call_command(payload)
    except BaseException as exc:
        return (
            2 if _fail_callback(_strip(exc), "Python Callback failed", err, size) else 0
        )
    _bytes(out, str(result).encode("utf-8", "replace"))
    return 1


@_hook("call_number", c_int, c_void_p, c_int, c_int, ctypes.POINTER(c_double), *ERR)
def call_number(payload, narg, report, x, err, size):
    """2: it failed, and the failure is only for the caller to know."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        args = _pop_arguments(narg)
    except _HocError as exc:
        _write(err, size, str(exc))
        return 0
    except BaseException as exc:
        return _fail(_strip(exc), "not a Python string", EXCEPTION, err, size)
    try:
        result = payload(*args)
    except BaseException as exc:
        if report:
            return _fail(_strip(exc), "func_call failed", HOC_ERROR, err, size)
        return 2
    x[0] = float(result) if _is_hoc_number(result) else 0.0
    return 1


_vector_wrappers = {}


@_hook("praxis", c_int, c_void_p, c_void_p, ctypes.POINTER(c_double), *ERR)
def praxis(payload, vec, x, err, size):
    """3: float() of the result failed, which is no HOC error."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        result = payload(_to_python(vec))
    except BaseException as exc:
        _strip(exc)
        if not isinstance(exc, (Exception, SystemExit)):
            print_exception(exc)
            x[0] = 1e9
            return 1
        return _fail(
            exc, "Call of Python Callable failed in praxis_efun", HOC_ERROR, err, size
        )
    try:
        x[0] = float(result)
    except BaseException as exc:
        _strip(exc)
        if _is_returned() and _has_caller():
            _keep(exc)
        return 3
    return 1


@_hook("bind_arguments", c_int, c_void_p, c_int, ctypes.POINTER(c_void_p), *ERR)
def bind_arguments(payload, narg, result, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        args = _pop_arguments(narg)
    except _HocError as exc:
        _write(err, size, str(exc))
        return 0
    except BaseException as exc:
        return _fail(_strip(exc), "not a Python string", EXCEPTION, err, size)
    result[0] = _new_python_object((payload, tuple(args)))
    return 1


@_hook("call_with_section", c_int, c_void_p, c_void_p, *ERR)
def call_with_section(payload, sec, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        section = _section_for(sec)
    except BaseException as exc:
        return _fail_hook(_strip(exc), err, size)
    try:
        payload(section)
    except BaseException as exc:
        return _fail_callback(_strip(exc), "Call of Python Callable failed", err, size)
    return 1


@_hook("command_tool", c_int, c_void_p, c_int, c_double, c_double, c_int, *ERR)
def command_tool(payload, type_, x, y, key, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        payload(type_, x, y, key)
    except BaseException as exc:
        return _fail_callback(_strip(exc), "Python Callback failed", err, size)
    return 1


# ---- GUI variables: a (target, key) tuple


def _is_item(target, key):
    """A HOC object's names are attributes, though it has __getitem__."""
    if isinstance(target, OBJECT_BASE) and isinstance(key, str):
        return False
    return bool(_PySequence_Check(target) or _PyMapping_Check(target))


@_hook("variable_get", c_int, c_void_p, ctypes.POINTER(c_double), *ERR)
def variable_get(payload, x, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        target, key = payload[0], payload[1]
        value = target[key] if _is_item(target, key) else getattr(target, key)
        if isinstance(value, (str, bytes)):  # PyFloat_AsDouble refuses them
            raise TypeError(f"must be real number, not {type(value).__name__}")
        x[0] = float(value)
    except BaseException as exc:
        x[0] = -1.0
        return _fail_callback(_strip(exc), "Python GUI get failed", err, size)
    return 1


@_hook("variable_set", c_int, c_void_p, c_double, *ERR)
def variable_set(payload, x, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        target, key = payload[0], payload[1]
        if _is_item(target, key):
            target[key] = x
        else:
            setattr(target, key, x)
    except BaseException as exc:
        return _fail_callback(_strip(exc), "Python GUI set failed", err, size)
    return 1


@_hook("variable_string", c_int, c_void_p, c_void_p, *ERR)
def variable_string(payload, out, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        target, key = payload[0], payload[1]
        value = getattr(target, key)
    except BaseException as exc:
        return _fail_callback(_strip(exc), "Python GUI get str failed", err, size)
    try:
        _bytes(out, str(value).encode("utf-8"))
    except Exception:
        pass
    return 1


# ---- the GUI in Python

_gui = [None]


def _gui_argument(references, pointer):
    kind = _nrn_stack_type()
    if kind == _STR:
        string = _nrn_str_pop0()
        if references:
            return reference(ctypes.cast(string, c_void_p).value)
        return _decode_hoc_string(string)
    if kind in (_OBJVAR, _OBJTMP):
        obj = _nrn_object_pop()
        value = _wrap_owned_object(obj) if obj else None
        return reference(value) if pointer else value
    return _nrn_double_pop()


@_hook(
    "gui_call",
    c_int,
    c_char_p,
    c_void_p,
    c_void_p,
    c_int,
    c_int,
    c_int,
    c_int,
    ctypes.POINTER(c_void_p),
    c_void_p,
    *ERR,
)
def gui_call(
    name, obj, self, narg, pointers, references, want_string, result, string, err, size
):
    """2: there is no GUI callback, NEURON does it itself. A NULL name sets the
    callback (obj, a PythonObject, or NULL). The arguments of the HOC call are on
    the stack; bit i of pointers is set if argument i + 1 is a Pointer to a
    pointer argument."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    if name is None:
        _gui[0] = _to_python(obj) if obj else None
        return 1
    i = narg
    try:
        args = [None] * narg
        for i in range(narg - 1, -1, -1):
            args[i] = _gui_argument(references, i < 31 and pointers >> i & 1)
    except BaseException as exc:
        _discard_pops(i)
        return _fail_hook(_strip(exc), err, size)
    callback = _gui[0]
    if callback is None:
        return 2
    try:
        value = callback(name.decode(), _to_python(obj), _to_python(self), *args)
    except BaseException as exc:
        print_exception(_strip(exc))
        value = 0
    if want_string:
        if not isinstance(value, (str, bytes)):
            return 2
        _bytes(string, value.encode("utf-8") if isinstance(value, str) else value)
    else:
        result[0] = to_hoc(value) or None
    return 1


# ---- sections and locations

_NOT_LOCATION = (
    "not a Python nrn.Section",
    "not a Python nrn.Segment",
    "not a Python nrn.Segment, rxd.node, or other with a segment property",
)


@_hook(
    "location",
    c_int,
    c_void_p,
    c_int,
    ctypes.POINTER(c_void_p),
    ctypes.POINTER(c_double),
    *ERR,
)
def location(payload, kind, sec, x, err, size):
    """kind: 0 a section, 1 a segment, 2 a segment or what has one (rxd's node)."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        if kind == 0:
            sec[0] = _sec_address(payload) if isinstance(payload, SECTION_BASE) else 0
            if not sec[0]:
                raise _HocError(_NOT_LOCATION[0])
            return 1
        if not isinstance(payload, SEGMENT_BASE):
            if kind == 1:
                raise _HocError(_NOT_LOCATION[1])
            if isinstance(payload, list):
                if len(payload) != 1:
                    raise _HocError("If a list is supplied, it must be of length 1")
                payload = payload[0]
            try:
                payload = payload.segment
            except AttributeError:
                payload = None
            if not isinstance(payload, SEGMENT_BASE):
                raise _HocError(_NOT_LOCATION[2])
        wrapper = SEGMENT_BASE._sec.__get__(payload)
        address = _sec_address(wrapper) if isinstance(wrapper, SECTION_BASE) else 0
        if not address or not _nrn_section_is_active(address):
            raise _HocError("nrn.Segment associated with deleted internal Section")
        x[0] = SEGMENT_BASE._x.__get__(payload)
        sec[0] = address
    except _HocError as exc:
        _write(err, size, str(exc))
        return 0
    except BaseException as exc:
        return _fail_hook(_strip(exc), err, size)
    return 1


def _sec_address(wrapper):
    try:
        return SECTION_BASE._sec.__get__(wrapper) or 0
    except AttributeError:
        return 0


@_hook("is_segment", c_int, py_object)
def is_segment(payload):
    return int(isinstance(payload, SEGMENT_BASE))


@_hook("segment_new", c_int, c_void_p, c_double, ctypes.POINTER(c_void_p), *ERR)
def segment_new(sec, x, result, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        result[0] = _new_python_object(Segment(_section_for(sec), x))
    except BaseException as exc:
        return _fail_hook(_strip(exc), err, size)
    return 1


@_hook("sections", c_int, c_void_p, c_void_p, *ERR)
def sections(payload, out, err, size):
    """The sections of an iterable, also those before a failure (NEURON appends
    them first)."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        iterator = iter(payload)
    except Exception:
        _write(err, size, "argument must be an iterable")
        return 0
    except BaseException as exc:
        return _fail_hook(_strip(exc), err, size)
    addresses = []
    status = 1
    try:
        for item in iterator:
            address = _sec_address(item) if isinstance(item, SECTION_BASE) else 0
            if not address:
                _write(err, size, "iterable must contain only Section objects")
                status = 0
                break
            addresses.append(address)
    except BaseException as exc:
        status = _fail(
            _strip(exc), "argument must be a Python iterable", EXCEPTION, err, size
        )
    _bytes(out, (c_void_p * len(addresses))(*addresses))
    return status


# The names and cells of the sections Python made: the owner is the Section,
# which may be on its way out (no reference is taken to it here).


def _owner(owner, slot):
    try:  # also a Section of this interface imported before
        return object.__getattribute__(_object_at(owner), slot)
    except AttributeError:
        return None


@_hook("section_name", c_int, c_void_p, c_void_p)
def section_name(owner, out):
    """0: it has none, and is __nrnsec_<address>."""
    name = _owner(owner, "_name")
    if name is None:
        return 0
    _bytes(out, name)
    return 1


def _cell(owner):
    weak = _owner(owner, "_cell")
    try:
        return None if weak is None else weak()
    except Exception:
        return None


@_hook("section_cell", c_int, c_void_p, ctypes.POINTER(c_void_p))
def section_cell(owner, result):
    try:
        result[0] = to_hoc(_cell(owner)) or None
    except BaseException as exc:
        print_exception(_strip(exc))
        return 0
    return 1


@_hook("section_cell_equals", c_int, c_void_p, c_void_p)
def section_cell_equals(owner, obj):
    cell = _cell(owner)
    if isinstance(cell, OBJECT_BASE):
        return int(_wrapped(cell) == (obj or 0))
    if not obj:
        return int(cell is None)
    held = c_void_p()
    return int(bool(_payload_of(obj, ctypes.byref(held))) and held.value == id(cell))


# ---- Vector


def _array_interface(source):
    """The address and stride of native doubles in an object with
    __array_interface__, such as a numpy array, or None."""
    try:
        interface = source.__array_interface__
        if interface["typestr"] != ("<f8" if sys.byteorder == "little" else ">f8"):
            return None
        address = interface["data"][0]
        strides = interface.get("strides")
        stride = strides[0] if strides else 8
        return int(address), int(stride)
    except Exception:
        return None


@_hook("numbers", c_int, c_char_p, c_void_p, c_void_p, *ERR)
def numbers(name, source, out, err, size):
    """Vector.from_python: the doubles of a sequence or an iterable."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    source = _payload(source)
    try:
        if not _PySequence_Check(source):
            try:
                items = list(source)
            except BaseException as exc:
                return _fail(
                    _strip(exc),
                    name.decode()
                    + " does not support the Python Sequence or Iterator protocol",
                    HOC_ERROR,
                    err,
                    size,
                    False,
                )
        elif isinstance(source, list):
            items = source
        else:
            interface = _array_interface(source)
            if interface is not None:
                n = len(source)
                address, stride = interface
                if stride == 8:
                    data = ctypes.string_at(address, 8 * n)
                else:
                    data = b"".join(
                        ctypes.string_at(address + i * stride, 8) for i in range(n)
                    )
                _bytes(out, data)
                return 1
            items = source
        n = len(items)
        values = (c_double * n)()
        for i in range(n):
            item = items[i]
            if item.__class__ is float:
                values[i] = item
                continue
            try:
                values[i] = item  # PyFloat_AsDouble
            except Exception:
                _write(err, size, f"item {i} is not a valid number")
                return 0
    except BaseException as exc:
        return _fail_hook(_strip(exc), err, size)
    _bytes(out, values)
    return 1


@_hook(
    "vector_to_python",
    c_int,
    ctypes.POINTER(c_double),
    c_size_t,
    c_char_p,
    c_void_p,
    ctypes.POINTER(c_void_p),
    *ERR,
)
def vector_to_python(data, n, name, held, result, err, size):
    """Vector.to_python: a new list, or into the sequence a PythonObject holds."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    if not held:
        result[0] = _new_python_object(data[:n])
        return 1
    target = _payload(held)
    if not _PySequence_Check(target):
        _write(err, size, name.decode() + "  is not a Python Sequence")
        return 0
    try:
        same = len(target) == n
    except Exception:
        same = False
    if not same:
        _write(err, size, name.decode() + " Python Sequence not same size as Vector")
        return 0
    interface = _array_interface(target)
    if interface is not None:
        address, stride = interface
        if stride == 8:
            ctypes.memmove(address, data, 8 * n)
        else:
            for i in range(n):
                ctypes.cast(address + i * stride, _DOUBLE_P)[0] = data[i]
        return 1
    for i in range(n):
        try:
            target[i] = data[i]
        except BaseException as exc:
            return _fail(
                _strip(exc),
                f"Could not set a Python Sequence item {i} of {n}",
                HOC_ERROR,
                err,
                size,
            )
    return 1


@_hook("vector_as_numpy", c_int, c_void_p, c_size_t, ctypes.POINTER(c_void_p), *ERR)
def vector_as_numpy(data, n, result, err, size):
    """The array shares the Vector's data."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        import numpy
        import numpy.ctypeslib

        if n:
            array = numpy.ctypeslib.as_array(ctypes.cast(data, _DOUBLE_P), (n,))
        else:
            array = numpy.empty(0)
    except BaseException as exc:
        return _fail(_strip(exc), "Vector.as_numpy() error", EXCEPTION, err, size)
    result[0] = _new_python_object(array)
    return 1


# ---- ParallelContext. 3: an error as NEURON's Python module threw it, which a
# method's caller makes "Class[i]::method: Python exception"; 4: printed, as
# std::bad_cast.


def _pickling_failed(exc, keep=False):
    _strip(exc)
    if keep and _is_returned() and _has_caller():
        _keep(exc)
    return 3


@_hook("pickle", c_int, c_void_p, c_void_p, *ERR)
def pickle_(payload, out, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    try:
        data = pickle.dumps(payload)
    except BaseException as exc:
        return _pickling_failed(exc)
    _bytes(out, data)
    return 1


@_hook("unpickle", c_int, c_void_p, c_size_t, ctypes.POINTER(c_void_p), *ERR)
def unpickle(data, n, result, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        value = pickle.loads(ctypes.string_at(data, n))
    except BaseException as exc:
        return _pickling_failed(exc)
    result[0] = _new_python_object(value)
    return 1


@_hook("call_pickled", c_int, c_void_p, c_size_t, c_int, c_void_p, *ERR)
def call_pickled(data, n, narg, out, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        f = pickle.loads(ctypes.string_at(data, n))
    except BaseException as exc:
        return _pickling_failed(exc)
    try:
        args = _pop_arguments(narg)
    except _HocError as exc:
        _write(err, size, str(exc))
        return 0
    except BaseException as exc:
        return _fail(_strip(exc), "not a Python string", EXCEPTION, err, size)
    try:
        result = f(*args)
    except BaseException as exc:
        _strip(exc)
        if not isinstance(exc, Exception):
            # As NEURON's Python module: printed, then pickling no result fails.
            print_exception(exc)
            return 4
        return _error_traceback(exc, "PyObject method call failed:", err, size)
    try:
        pickled = pickle.dumps(result)
    except BaseException as exc:
        return _pickling_failed(exc)
    _bytes(out, pickled)
    return 1


# ---- ParallelContext collectives: py_alltoall (1), py_allgather (2), py_gather (3),
# py_broadcast (4), py_scatter (5). The pickles travel as the bytes of HOC's
# pc.alltoall. A rank that sends nothing to another sends None.


def _exchange(pc, chunks):
    """Send chunks[i] (bytes) to rank i; the bytes each rank sent here."""
    from . import n

    nhost = len(chunks)
    sent = n.Vector([len(c) for c in chunks])
    received = n.Vector()
    pc.alltoall(sent, n.Vector(nhost, 1), received)
    data = n.Vector()
    pc.alltoall(n.Vector(list(b"".join(chunks))), sent, data)
    data = bytes(int(x) for x in data)
    out, start = [], 0
    for count in received:
        out.append(data[start : start + int(count)])
        start += int(count)
    return out


class _PickleFailed(Exception):
    """pickle failed in a collective: NEURON's C++ exception "Python exception"."""


def _dumps(value):
    try:
        return pickle.dumps(value)
    except BaseException as exc:
        raise _PickleFailed() from exc


def _loads(data):
    try:
        return pickle.loads(data) if data else None
    except BaseException as exc:
        raise _PickleFailed() from exc


def _items(source):
    """The pickled items of a list, one per rank, None as nothing (PyList_GetItem)."""
    return [b"" if x is None else _dumps(x) for x in list.__iter__(source)]


def _collective(kind, size, source):
    from . import n

    pc = n.ParallelContext()
    nhost, rank = int(pc.nhost()), int(pc.id())
    if kind == 1 or (kind == 5 and rank == size):
        if not isinstance(source, list):
            raise _HocError("Argument must be a Python list")
        if list.__len__(source) != nhost:
            which = "py_alltoall" if kind == 1 else "py_scatter"
            raise _HocError(which + " list size must be nhost")
    if nhost == 1:  # repackaged, not pickled
        if kind == 5:
            return list.__getitem__(source, 0)
        return [source] if kind in (2, 3) else source
    if kind not in (1, 2) and not 0 <= size < nhost:
        raise _HocError("root rank must be >= 0 and < nhost")
    if kind == 1:
        chunks = _items(source)
        if size < 0:  # only the numbers of bytes to be sent and received
            received = n.Vector()
            pc.alltoall(
                n.Vector([len(c) for c in chunks]), n.Vector(nhost, 1), received
            )
            return (sum(len(c) for c in chunks), int(sum(received)))
        return [_loads(c) for c in _exchange(pc, chunks)]
    if kind == 2:
        return [_loads(c) for c in _exchange(pc, [_dumps(source)] * nhost)]
    root = size
    if kind == 3:
        data = _dumps(source)
        received = _exchange(pc, [data if i == root else b"" for i in range(nhost)])
        return [_loads(c) for c in received] if rank == root else None
    if kind == 4:
        data = _dumps(source) if rank == root else b""
        received = _exchange(pc, [b"" if i == root else data for i in range(nhost)])
        return source if rank == root else _loads(received[root])
    chunks = _items(source) if rank == root else [b""] * nhost
    return _loads(_exchange(pc, chunks)[root])


@_hook("collective", c_int, c_int, c_int, c_int, ctypes.POINTER(c_void_p), *ERR)
def collective(kind, size, pointer, result, err, errsize):
    """The argument of the HOC call is on the stack (a Pointer to a pointer
    argument if pointer). size is the root rank for 3, 4 and 5; for 1, a negative
    size asks for the numbers of bytes only. 3: pickling failed, as NEURON's
    "Python exception"."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        source = _gui_argument(False, pointer)
    except BaseException as exc:
        return _fail_hook(_strip(exc), err, errsize)
    try:
        value = _collective(kind, size, source)
    except _HocError as exc:
        _write(err, errsize, str(exc))
        return 0
    except _PickleFailed as exc:
        return _pickling_failed(exc.__cause__)
    except BaseException as exc:
        return _fail_hook(_strip(exc), err, errsize)
    result[0] = to_hoc(value) or None
    return 1


# ---- SaveState, CoreNEURON, RangeVarPlot. 2: nothing registered.

_savestate = [None, None]
_rvp = [None]


def set_savestate_callbacks(store, restore, replace=False):
    """Register the Python state SaveState keeps, for packages such as neuron.rxd.

    ``store()`` returns a bytes-like object and ``restore(bytearray)`` receives
    it back. A failure is the HOC error of the SaveState call. ``replace`` is
    accepted for the released core's signature: here a call always replaces.
    """
    _savestate[:] = store, restore
    return True


def set_rvp_converter(converter):
    """Register the function that turns a RangeVarPlot's Python object, such as
    an rxd species, into a function of x; it returns anything else unchanged."""
    _rvp[0] = converter


@_hook("savestate_store", c_int, c_void_p, *ERR)
def savestate_store(out, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    store = _savestate[0]
    if store is None:
        return 2
    try:
        state = store()
        if isinstance(state, int):  # PyBytes_FromObject refuses it
            raise TypeError(f"cannot convert '{type(state).__name__}' object to bytes")
        data = bytes(state)
    except BaseException as exc:
        return _fail(
            _strip(exc), "SaveState: Data store failure.", HOC_ERROR, err, size
        )
    _bytes(out, data)
    return 1


@_hook("savestate_restore", c_int, c_void_p, c_size_t, *ERR)
def savestate_restore(data, n, err, size):
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    restore = _savestate[1]
    if restore is None:
        return 2
    try:
        restore(bytearray(ctypes.string_at(data, n)))
    except BaseException as exc:
        return _fail(
            _strip(exc), "SaveState: Data restore failure.", HOC_ERROR, err, size
        )
    return 1


@_hook("coreneuron_option", c_int, c_int, ctypes.POINTER(c_int))
def coreneuron_option(which, value):
    """CoreNEURON's options are read while the simulator is set up: an error is
    printed, and the option is -1."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        option = (
            _coreneuron.enable_value if which == 0 else _coreneuron.file_mode_value
        )()
        value[0] = operator.index(option)
    except BaseException as exc:
        print_exception(_strip(exc))
        value[0] = -1
        return 0
    return 1


@_hook("coreneuron_arg", c_int, c_double, c_void_p)
def coreneuron_arg(tstop, out):
    """0: no arguments."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    try:
        arg = _coreneuron.nrncore_arg(tstop)
    except BaseException as exc:
        print_exception(_strip(exc))
        return 0
    if arg is None:
        return 0
    _bytes(out, str(arg).encode("utf-8"))
    return 1


@_hook("rvp_callable", c_int, c_void_p, ctypes.POINTER(c_void_p), *ERR)
def rvp_callable(payload, result, err, size):
    """2: plot the object itself."""
    EPOCH[0] = None  # NEURON calls Python: HOC may have changed the model
    payload = _payload(payload)
    converter = _rvp[0]
    if converter is None:
        return 2
    try:
        value = converter(payload)
    except BaseException as exc:
        print_exception(_strip(exc))
        return 2
    if value is None:
        return 2
    result[0] = to_hoc(value) or None
    return 1


# ---- Windows: a thread that waits for the GUI's, which calls Python and HOC,
# releases the HOC lock as often as it holds it, and the GIL, and takes them again.

_WAIT = ctypes.CFUNCTYPE(None, c_void_p)  # not a PYFUNCTYPE: the call releases the GIL


@_hook("wait", None, c_void_p, c_void_p)
def wait(function, arg):
    count = 0
    while True:
        try:
            HOC_LOCK.release()
        except RuntimeError:  # it does not hold the lock any more
            break
        count += 1
    try:
        _WAIT(function)(arg)
    finally:
        for _ in range(count):
            HOC_LOCK.acquire()


def install():
    """Register every hook with NEURON, and clear them when Python exits."""
    error = ctypes.create_string_buffer(256)
    for name, callback in _HOOKS.items():
        if _hook_set(name.encode(), ctypes.cast(callback, c_void_p), error, len(error)):
            raise ImportError(f"NEURON refuses the Python hook {name}: {error.value!r}")
    import atexit

    atexit.register(uninstall)


def uninstall():
    for name in _HOOKS:
        _hook_set(name.encode(), None, None, 0)
