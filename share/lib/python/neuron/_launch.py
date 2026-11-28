"""How nrniv and special run Python: scripts, -c, the prompt (src/nrniv/nrnpy_launch.cpp).

Not imported as neuron._launch, which would import neuron: the launcher runs this
file in a namespace of its own, where it has set executable, argv and neuron_parent.
"""

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


def flush():
    # stdout is buffered when it is not a terminal; HOC's output comes next.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def import_readline():
    # Imported later, it would replace NEURON's PyOS_ReadlineFunctionPointer.
    try:
        import readline
    except ImportError:
        pass


def interact():
    # The console is the standard library's: a code.py in the working directory must not replace
    # it, also when PYTHONPATH names that directory (an empty entry, as in PYTHONPATH=:dir).
    here = os.path.realpath(os.getcwd())
    path = sys.path[:]
    sys.path[:] = [p for p in basic_path if os.path.realpath(p or os.curdir) != here]
    try:
        import code
    finally:
        sys.path[:] = path
    code.InteractiveConsole(vars(sys.modules["__main__"])).interact("", exitmsg="")


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


sys.meta_path.append(NeuronModules)
set_path("")
