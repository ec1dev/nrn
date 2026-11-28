#pragma once
// The adapters between HOC and the interface's Python hooks (nrnpy_hooks.cpp). No CPython here: a
// hook is a C function the interface registers (nrn_python_hook_set), and a Python object is a
// void* that only the interface reads.
struct Object;

namespace neuron::python {

/// Make the HOC class PythonObject, and install the functions with which HOC and the simulator call
/// Python. Python itself starts when one of them needs it.
void register_python_object();

bool is_python_object(Object* o);
/// What a PythonObject holds, borrowed; null for HOC's `new PythonObject()` (Python's __main__).
void* payload(Object* o);
/// A HOC error unless o is a PythonObject and the interface's hooks are there.
void require_python_object(Object* o, const char* message);

/// While Python runs for HOC: HOC is at its top level (Python code that a template's method runs
/// calls HOC as any other).
class Top {
    Object* object_;
    int exceptions_{};
    union Objectdata* data_{};
    struct Symlist* symbols_{};
    int local_{};

  public:
    Top();
    Top(const Top&) = delete;
    Top& operator=(const Top&) = delete;
    ~Top();
};

}  // namespace neuron::python
