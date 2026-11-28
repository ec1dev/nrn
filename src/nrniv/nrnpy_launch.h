#pragma once
// Between nrniv -python's CPython part (nrnpy_launch.cpp) and the rest of NEURON (nrnpy.cpp).
#include <string>

extern int nrn_global_argc;
extern char** nrn_global_argv;

namespace neuron::python {
/// Bind the CPython functions in library (nrnpy_open_libpython), start the interpreter, and run
/// neuron/_launch.py (nrnpy_launch.cpp). Throws if it cannot.
void start_interpreter(void* library);

// nrnpy.cpp
/// A dlopen handle of the libpython to run, null if libnrniv is linked to it. Throws if there is
/// none.
void* nrnpy_open_libpython();
/// What Python takes as its executable: an absolute path. Throws if it is not.
std::string python_executable();
/// Would python run in UTF-8 mode here?
bool python_utf8_mode();
/// The directory with the neuron package of this NEURON, or "".
std::string neuron_package_parent();
}  // namespace neuron::python
