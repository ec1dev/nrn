"""How nrniv and special run Python: scripts, -c, the prompt, nrnpython() before
the neuron package is imported (src/nrniv/nrnpy_launch.cpp and nrnpy.cpp).

Not imported as neuron._launch, which would import neuron: the launcher runs this
file in a namespace of its own, where it has set executable, argv and neuron_parent.
"""

import atexit
import ctypes
import importlib.machinery
import os
import sys

sys.executable = executable  # noqa: F821
sys.argv = argv  # noqa: F821
basic_path = tuple(sys.path)


def set_path(filename):
    # As python: a script's directory, links resolved, or '' for anything else.
    first = os.path.dirname(os.path.realpath(filename)) if filename else ""
    sys.path[:] = (first, *basic_path)


def compile_file(filename):
    set_path(filename)
    with open(filename, "rb") as file:
        code = compile(file.read(), filename, "exec", dont_inherit=True)
    sys.modules["__main__"].__file__ = filename
    return code


def main_namespace():
    return vars(sys.modules["__main__"])


def flush():
    # stdout is buffered when it is not a terminal; HOC's output comes next.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def run_file(filename):
    try:
        exec(compile_file(filename), main_namespace())
    finally:
        flush()


def run_command(command):
    # nrniv -python -c command
    set_path("")
    exec(compile(command, "<string>", "exec", dont_inherit=True), main_namespace())


def interact(_):
    # The console is the standard library's: a code.py in the working directory must not replace
    # it, also when PYTHONPATH names that directory (an empty entry, as in PYTHONPATH=:dir).
    here = os.path.realpath(os.getcwd())
    path = sys.path[:]
    sys.path[:] = [p for p in basic_path if os.path.realpath(p or os.curdir) != here]
    try:
        import code
    finally:
        sys.path[:] = path
    code.InteractiveConsole(main_namespace()).interact("", exitmsg="")


def import_neuron(_):
    # HOC needs the interface: its hooks replace the statement hook below.
    import neuron


_READER = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
)
_kept = []


def prompt(_):
    # The prompt and input() read lines with NEURON's reader, which runs its events while it
    # waits. Neither name is in the stable ABI (docs/scripting.rst): without them Python's own
    # reader stays. readline, imported later, would replace NEURON's.
    try:
        import readline
    except ImportError:
        pass
    try:
        pointer = ctypes.c_void_p.in_dll(
            ctypes.pythonapi, "PyOS_ReadlineFunctionPointer"
        )
        malloc = ctypes.pythonapi.PyMem_RawMalloc
    except (ValueError, AttributeError):
        return
    malloc.argtypes, malloc.restype = [ctypes.c_size_t], ctypes.c_void_p
    # Not a PyDLL: the GIL is released while NEURON waits for a line.
    read = ctypes.CDLL(_LIBNRNIV).nrn_hoc_readline
    read.argtypes, read.restype = [ctypes.c_void_p], ctypes.c_char_p

    @_READER
    def reader(stdin, stdout, prompt):
        line = read(prompt)
        if line is None:
            return None  # an interrupt
        copy = malloc(len(line) + 1)  # Python frees it, with PyMem_RawFree
        if copy:
            ctypes.memmove(copy, line + b"\0", len(line) + 1)
        return copy

    _kept.append(reader)
    pointer.value = ctypes.cast(reader, ctypes.c_void_p).value


def _report(exc):
    # As PyErr_Print: SystemExit ends the process, anything else goes to sys.excepthook, without
    # the frames of this file.
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is not None and not isinstance(code, int):
            print(code, file=sys.stderr)
            code = 1
        ctypes.pythonapi.Py_Exit(code or 0)
    tb = exc.__traceback__
    while (
        tb is not None and tb.tb_frame.f_code.co_filename == launch_file
    ):  # noqa: F821
        tb = tb.tb_next
    sys.last_exc, sys.last_type, sys.last_value, sys.last_traceback = (
        exc,
        type(exc),
        exc,
        tb,
    )
    sys.excepthook(type(exc), exc, tb)


_ACTIONS = {
    "run_file": run_file,
    "run_command": run_command,
    "interact": interact,
    "prompt": prompt,
    "set_path": set_path,
    "import_neuron": import_neuron,
}


@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
def _launcher(what, arg):
    # What nrniv asks of Python (src/nrniv/nrnpy.cpp). 0: it failed, and that is printed.
    try:
        _ACTIONS[what.decode()](None if arg is None else decode(arg))  # noqa: F821
    except BaseException as exc:
        _report(exc)
        return 0
    return 1


@ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.c_size_t,
)
def _statement(text, ok, err, size):
    # nrnpython(statement) before the neuron package is imported: in __main__, and an error
    # printed as python prints it.
    try:
        exec(compile(text, "<string>", "exec", dont_inherit=True), main_namespace())
        ok[0] = 1
    except BaseException as exc:
        _report(exc)
    return 1


class NeuronModules:
    # hoc, nrn and _neuron_section, which the neuron package makes, before it is imported.
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name in ("hoc", "nrn", "_neuron_section") and "neuron" not in sys.modules:
            import neuron

            return importlib.machinery.ModuleSpec(name, NeuronModules)

    @staticmethod
    def create_module(spec):
        return sys.modules[spec.name]

    @staticmethod
    def exec_module(module):
        pass


# libnrniv is loaded already: nrniv and special link it.
_LIBNRNIV = "libnrniv.dll" if os.name == "nt" else None
_hook_set = ctypes.PyDLL(_LIBNRNIV).nrn_python_hook_set
_hook_set.argtypes = [
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
]
for _name, _hook in (b"statement", _statement), (b"launcher", _launcher):
    _hook_set(_name, ctypes.cast(_hook, ctypes.c_void_p), None, 0)
    atexit.register(_hook_set, _name, None, None, 0)  # before the functions go
sys.meta_path.append(NeuronModules)
set_path("")
