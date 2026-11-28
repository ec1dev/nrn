#include <../../nrnconf.h>

#include "nrnpy.h"
#include "nrnpy_capi.h"

#include <thread>

extern NRN_DLLSYM void (*p_nrnpython_finalize)();
extern "C" NRN_DLLSYM void nrnpy_set_pr_etal(int (*cbpr_stdoe)(int, char*));

#ifndef NRNPYTHON_DYNAMICLOAD
// libnrniv is linked to libpython, whose functions are found as when they are loaded: this use
// keeps the link.
extern "C" int Py_IsInitialized(void);
extern "C" NRN_EXPORT int (*nrnpy_linked)(void) = Py_IsInitialized;
#endif

namespace neuron::python {

Capi py;
bool (*launch)();

namespace {
std::thread::id host_thread;
bool bound{};

}  // namespace

bool Capi::bind(void* library, std::string& missing) {
    // The first name that is missing, if one is.
#define X(ret, name, args)                                                        \
    if (!(name = reinterpret_cast<ret(*) args>(python_symbol(library, #name)))) { \
        missing = missing.empty() ? #name : missing;                              \
    }
    NRNPY_FUNCTIONS(X)
#undef X
#define X(name)                                                            \
    if (!(name = static_cast<PyObject*>(python_symbol(library, #name)))) { \
        missing = missing.empty() ? #name : missing;                       \
    }
    NRNPY_TYPES(X)
#undef X
    return missing.empty();
}

// HOC's quit(): finalize Python, in the thread that brought NEURON and Python together. Python
// waits for the others.
void finalize() {
    if (std::this_thread::get_id() == host_thread) {
        py.PyGILState_Ensure();  // nothing is left to release it to
        py.PyErr_Clear();        // quit() in Python code that is handling an error
        clear_hooks();           // their functions go with Python
        py.Py_FinalizeEx();
    }
}

bool start() {
    if (bound) {
        return py.Py_IsInitialized();
    }
    host_thread = std::this_thread::get_id();
    std::string missing;
    // Python is the main program: it loaded libnrniv, or a program that embeds it did.
    if (py.bind(nullptr, missing) && py.Py_IsInitialized()) {
        p_nrnpython_finalize = finalize;
    } else if (!launch || !launch()) {
        return false;
    }
    bound = true;
    // The interface's output function goes with Python, but NEURON still prints on its way out.
    py.Py_AtExit([] { nrnpy_set_pr_etal(nullptr); });
    return true;
}

}  // namespace neuron::python
