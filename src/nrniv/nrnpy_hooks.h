#pragma once
#include <exception>

#include "nrnpy_capi.h"

struct Object;
union Objectdata;
struct Symlist;

namespace neuron::python {

/// Make the HOC class PythonObject, and install the functions with which HOC and the simulator call
/// Python. Python itself starts when one of them needs it.
void register_python_object();
/// Python is about to be finalized: no hook is called any more.
void clear_hooks();

bool is_python_object(Object* o);
/// What a PythonObject holds, borrowed; __main__ for HOC's `new PythonObject()`. The GIL is held.
PyObject* payload(Object* o);
/// A HOC error unless o is a PythonObject and Python runs.
void require_python_object(Object* o, const char* message);

/// The model may have changed: what the interface keeps about it is out of date. The GIL is held.
void model_changed();

/// Entering Python: hold the GIL, with HOC at its top level (Python code that a template's method
/// runs calls HOC as any other), and what the interface keeps about the model is out of date (HOC
/// may have edited it since Python ran).
class Top {
    Gil gil_;
    Object* object_;
    int exceptions_{};
    Objectdata* data_{};
    Symlist* symbols_{};
    int local_{};

  public:
    Top();
    Top(const Top&) = delete;
    Top& operator=(const Top&) = delete;
    ~Top();
};

/// A C callback (hoccommand_exec, Vector.apply) failed with the Python error that is set: the
/// interface prints it, or it is the HOC error. Returns if it was only printed.
void callback_failed(int kind);

}  // namespace neuron::python
