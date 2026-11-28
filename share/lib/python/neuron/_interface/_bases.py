"""The base classes of the wrappers on a core that calls Python (api.CALLS_PYTHON).

NEURON's base types (src/nrniv/nrnpy_fast.cpp, nrn_python_types) hold what a
wrapper stands for, ``_obj`` or ``_sec``, and release it when the wrapper goes:
their tp_dealloc runs after a subclass's ``__del__`` and after the collector,
and neither can skip it. The other slots are these classes'. Segment, Mechanism
and Vector also derive from the core's fast path types, unless it is off
(NRN_PYTHON_FAST_PATH=0); fast_setup() gives it their classes.
"""

import ctypes

from . import api as _api

if _api.CALLS_PYTHON:
    _types = ctypes.PYFUNCTYPE(ctypes.py_object)(("nrn_python_types", _api.libnrniv))
    _fast_setup = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)(
        ("nrn_python_fast_setup", _api.libnrniv)
    )
    _section_new_owned = _api._bind(
        "nrn_section_new_owned_nothrow",
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    )
    # The last three are the fast path's: None when it is off.
    _OBJECT, _SECTION, _SEGMENT, MECHANISM_BASE, VECTOR_BASE = _types()
else:  # a released core: _released_base.py has the bases, these are not used
    _OBJECT = _SECTION = object
    _SEGMENT = MECHANISM_BASE = VECTOR_BASE = None


class OBJECT_BASE(_OBJECT):
    __slots__ = ("_class_name", "_methods", "_section_cache", "_plotshape_var")


class SECTION_BASE(_SECTION):
    """Sections made in Python are owned by their objects (nrn_section_new_owned)."""

    __slots__ = ("_name", "_cell", "_inserted_mechs", "_inserted_mechs_epoch")

    def __init__(self, name, cell):
        # name is bytes (with the cell's) or None; cell a weak reference or None.
        try:
            SECTION_BASE._sec.__get__(self)
        except AttributeError:
            pass
        else:
            return  # made before, as NPySecObj_init
        put = object.__setattr__
        put(self, "_name", name)
        put(self, "_cell", cell)
        put(self, "_inserted_mechs", None)
        put(self, "_inserted_mechs_epoch", None)
        sec, error = ctypes.c_void_p(), _api._ERR_BUF()
        if _section_new_owned(id(self), sec, error, len(error)):  # data frozen, for one
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        put(self, "_sec", sec.value)
        put(self, "_pynative", True)


# A slot write that the fast path's setattro, which object.__setattr__ refuses to bypass, makes.
_put = (_SEGMENT or object).__setattr__


class SEGMENT_BASE(*(() if _SEGMENT is None else (_SEGMENT,))):
    __slots__ = ("_sec", "_x", "_mech_cache")
    if _SEGMENT is not None:
        # Filled by sections.py and mechanism.py once a name has been resolved:
        # name -> Symbol address of a scalar range variable, and mechanism name
        # -> {unsuffixed name -> Symbol address}.
        _fast_rangevars = {}
        _fast_mechanisms = {}

    def __init__(self, sec, x):
        _put(self, "_sec", sec)
        _put(self, "_x", x)
        _put(self, "_mech_cache", None)


def fast_setup():
    """Give the fast path the interface's classes and tables, once they exist."""
    if _SEGMENT is None:
        return
    base = SEGMENT_BASE
    segment, mechanism = base._fast_segment_class, base._fast_mechanism_class
    objects = (
        segment,
        mechanism,
        base._fast_vector_class,
        base._fast_epoch,
        base._fast_nseg_generation,
        base._fast_rangevars,
        base._fast_mechanisms,
        0,  # a section's nseg generation until it has one
        AttributeError,
        segment.__basicsize__,  # it finds the slots in instances of this size
        mechanism.__basicsize__,
    )
    if _fast_setup(objects):
        raise ImportError("NEURON's fast path refuses the interface's classes")
