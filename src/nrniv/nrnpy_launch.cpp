// Start CPython under nrniv or special, and hand over to neuron/_launch.py. It does the rest
// (scripts, -c, the prompt, nrnpython() before the neuron package is imported) through a hook it
// registers as the interface registers its own (nrn_python_hook_set).
#include <../../nrnconf.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "nrnpy.h"
#include "nrnpy_capi.h"
#include "nrnpy_launch.h"
#include "utils/logger.hpp"

namespace neuron::python {
namespace {

// Finds neuron/_launch.py without importing neuron, and runs it in a namespace of its own, where
// args are the executable, the directory of the neuron package and argv, as main() got them: bytes,
// each with a NUL at its end.
constexpr auto bootstrap = R"py(
import importlib.util, os, sys
decode = (lambda b: b.decode("mbcs", "surrogateescape")) if os.name == "nt" else os.fsdecode
executable, neuron_parent, *argv = map(decode, args.split(b"\0")[:-1])
if neuron_parent and neuron_parent not in sys.path:
    sys.path.append(neuron_parent)
launch_file = os.path.join(os.path.dirname(importlib.util.find_spec("neuron").origin), "_launch.py")
with open(launch_file, "rb") as f:
    exec(compile(f.read(), launch_file, "exec", dont_inherit=True))
)py";

// PyConfig is not in the stable ABI. Py_Initialize reads these globals, if they are there
// (docs/scripting.rst).
bool set_flag(void* library, const char* name) {
    auto* const flag = static_cast<int*>(python_symbol(library, name));
    if (flag) {
        *flag = 1;
    }
    return flag;
}

void initialize(void* library) {
    if (nrnpy_nositeflag && !set_flag(library, "Py_NoSiteFlag")) {
        Fprintf(stderr,
                "Warning: -Py_NoSiteFlag is ignored. Py_Initialize() imports site, "
                "so sitecustomize and usercustomize still run.\n");
    }
    if (python_utf8_mode()) {
        set_flag(library, "Py_UTF8Mode");
    }
    std::string const executable = python_executable();
    // Before Py_Initialize on purpose: the locale is still "C", so what it cannot represent becomes
    // surrogates, which encode back to the original bytes whatever the filesystem encoding will be.
    // Py_PreInitialize is not in the stable ABI. test/pyinit/c_locale_pyexe.py.
    wchar_t* const program_name = py.Py_DecodeLocale(executable.c_str(), nullptr);
    if (!program_name) {
        throw std::runtime_error("Could not decode the Python executable path from the locale");
    }
    py.Py_SetProgramName(program_name);  // keeps the pointer
    py.Py_Initialize();
    std::string args = executable + '\0' + neuron_package_parent() + '\0';
    for (int i = 0; i < nrn_global_argc; ++i) {
        args.append(nrn_global_argv[i]).push_back('\0');
    }
    PyObject* const names =
        py.Py_BuildValue("{sN}", "args", py.PyBytes_FromStringAndSize(args.data(), args.size()));
    PyObject* const code = names ? py.Py_CompileString(bootstrap, "<nrniv>", stable::file_input)
                                 : nullptr;
    if (!code || !py.PyEval_EvalCode(code, names, names)) {
        py.PyErr_Print();
        throw std::runtime_error("Could not set up Python");
    }
}

}  // namespace

void start_interpreter(void* library) {
    std::string missing;
    if (!py.bind(library, missing)) {
        throw std::runtime_error("libpython has no " + missing);
    }
    initialize(library);
}

}  // namespace neuron::python
