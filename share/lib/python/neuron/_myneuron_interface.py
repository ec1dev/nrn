"""Experimental headless h/HOC bootstrap; no legacy-extension fallback."""

import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys


class _HeadlessImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"hoc", "nrn", "_neuron_section"} or (
            fullname.startswith("neuron.")
            and fullname not in {"neuron._myneuron_interface", "neuron._config_params"}
        ):
            raise ImportError(
                f"{fullname} is not supported by the myneuron h/HOC beta; "
                "use NEURON_PYTHON_INTERFACE=legacy in a fresh process"
            )
        return None


def populate(namespace):
    if sys.implementation.name != "cpython" or not (
        (3, 10) <= sys.version_info[:2] < (3, 14)
    ):
        raise ImportError("the myneuron beta requires CPython 3.10-3.13")
    existing = {"hoc", "neuron.hoc", "nrn", "_neuron_section"} & sys.modules.keys()
    if existing:
        raise ImportError(
            "myneuron cannot replace an initialized legacy binding; start a fresh process"
        )
    if os.environ.get("MYNEURON_USE_CFUNCTYPE", "1") == "0":
        raise ImportError(
            "the myneuron selector requires its standalone CFUNCTYPE provider"
        )
    if os.environ.get("MYNEURON_NOGUI", "1") != "1":
        raise ImportError("the myneuron selector supports headless execution only")
    if importlib.util.find_spec("myneuron") is None:
        raise ImportError(
            "NEURON_PYTHON_INTERFACE=myneuron requires a compatible myneuron installation; "
            "install the matching beta wheel or select legacy in a fresh process"
        )

    # Install before importing the provider: accidental submodule imports must
    # fail before loading another binding. Keep the guard if initialization fails;
    # partially initialized native state cannot be rolled back by Python imports.
    if not any(isinstance(finder, _HeadlessImports) for finder in sys.meta_path):
        sys.meta_path.insert(0, _HeadlessImports())
    os.environ["MYNEURON_NOGUI"] = "1"
    home = Path(namespace["__file__"]).parent / ".data" / "share" / "nrn"
    if home.is_dir():
        os.environ["NEURONHOME"] = str(home)
    from myneuron import h, n

    version = h.nrnversion(5)
    namespace.update(h=h, n=n, __version__=version, version=version)
    namespace["__all__"] = ["h", "n", "__version__"]
