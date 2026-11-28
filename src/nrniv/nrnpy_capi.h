#pragma once
// The CPython functions that the fast path (nrnpy_fast.cpp) and nrniv -python (nrnpy_launch.cpp)
// call; nothing else in NEURON calls CPython. Python.h is not included and libpython is not linked
// when Python is dynamic: the functions are looked up in the running interpreter. All of them are
// in the stable ABI (test/api/python_abi.cpp), so one libnrniv serves every CPython 3.11 and later.
#include <cstddef>
#include <string>

#include "nrnpy.h"

struct _object;
typedef _object PyObject;

namespace neuron::python {

// clang-format off
#define NRNPY_FUNCTIONS(X)                                                        \
    X(int, Py_IsInitialized, (void))                                              \
    X(void, Py_Initialize, (void))                                                \
    X(int, Py_FinalizeEx, (void))                                                 \
    X(int, Py_AtExit, (void (*)()))                                               \
    X(void, Py_SetProgramName, (const wchar_t*))                                  \
    X(wchar_t*, Py_DecodeLocale, (const char*, std::size_t*))                     \
    X(int, PyGILState_Ensure, (void))                                             \
    X(void, PyGILState_Release, (int))                                            \
    X(PyObject*, Py_CompileString, (const char*, const char*, int))               \
    X(PyObject*, PyEval_EvalCode, (PyObject*, PyObject*, PyObject*))              \
    X(PyObject*, Py_BuildValue, (const char*, ...))                               \
    X(PyObject*, PyBytes_FromStringAndSize, (const char*, std::ptrdiff_t))       \
    X(void, Py_IncRef, (PyObject*))                                               \
    X(void, Py_DecRef, (PyObject*))                                               \
    X(void, PyErr_Fetch, (PyObject**, PyObject**, PyObject**))                    \
    X(void, PyErr_Restore, (PyObject*, PyObject*, PyObject*))                     \
    X(PyObject*, PyErr_Occurred, (void))                                          \
    X(int, PyErr_ExceptionMatches, (PyObject*))                                   \
    X(void, PyErr_Print, (void))                                                  \
    X(void, PyErr_Clear, (void))                                                  \
    X(PyObject*, PyObject_CallMethod, (PyObject*, const char*, const char*, ...)) \
    X(PyObject*, PyObject_GenericGetAttr, (PyObject*, PyObject*))                 \
    X(int, PyObject_GenericSetAttr, (PyObject*, PyObject*, PyObject*))            \
    X(int, PyType_IsSubtype, (PyObject*, PyObject*))                              \
    X(PyObject*, PyType_FromSpec, (void*))                                        \
    X(void*, PyType_GetSlot, (PyObject*, int))                                    \
    X(PyObject*, PyFloat_FromDouble, (double))                                    \
    X(double, PyFloat_AsDouble, (PyObject*))                                      \
    X(PyObject*, PyLong_FromSsize_t, (std::ptrdiff_t))                            \
    X(std::ptrdiff_t, PyLong_AsSsize_t, (PyObject*))                              \
    X(void*, PyLong_AsVoidPtr, (PyObject*))                                       \
    X(const char*, PyUnicode_AsUTF8AndSize, (PyObject*, std::ptrdiff_t*))         \
    X(PyObject*, PyTuple_GetItem, (PyObject*, std::ptrdiff_t))                    \
    X(PyObject*, PyList_GetItem, (PyObject*, std::ptrdiff_t))                     \
    X(int, PyList_SetItem, (PyObject*, std::ptrdiff_t, PyObject*))                \
    X(PyObject*, PyDict_GetItem, (PyObject*, PyObject*))                          \
    X(void, PyDict_Clear, (PyObject*))

// Objects in the stable ABI.
#define NRNPY_TYPES(X)                                                            \
    X(_Py_NoneStruct) X(_Py_FalseStruct) X(PyUnicode_Type)      \
    X(PyFloat_Type) X(PyLong_Type) X(PyDict_Type)
// clang-format on

struct Capi {
#define X(ret, name, args) ret(*name) args{};
    NRNPY_FUNCTIONS(X)
#undef X
#define X(name) PyObject* name{};
    NRNPY_TYPES(X)
#undef X
    /// Fill the table from the libpython that library is a dlopen handle of or, if it is null, from
    /// the one this process has already loaded. False, with a message in missing, if there is none.
    bool bind(void* library, std::string& missing);
};
extern Capi py;

/// What else libnrniv copies of the stable ABI. test/api/python_abi.cpp checks it against Python.h.
namespace stable {
/// An instance of the base types (nrnpy_fast.cpp); refcnt and type are PyObject of a build with the
/// GIL, the start of every object.
struct Wrapper {
    std::ptrdiff_t refcnt;
    PyObject* type;
    PyObject* address;   // _obj or _sec: an int
    PyObject* pynative;  // _pynative of a section: True if Python made it
};
struct Member {  // PyMemberDef
    const char* name;
    int type;
    std::ptrdiff_t offset;
    int flags;
    const char* doc;
};
struct Slot {  // PyType_Slot
    int id;
    void* function;
};
struct Spec {  // PyType_Spec
    const char* name;
    int basicsize;
    int itemsize;
    unsigned int flags;
    Slot* slots;
};
// Py_tp_*, Py_mp_*, Py_T_OBJECT_EX, Py_file_input, Py_TPFLAGS_*
constexpr int tp_dealloc = 52, tp_free = 74, tp_getattro = 58, tp_setattro = 69, tp_members = 72,
              mp_subscript = 5, mp_ass_subscript = 3, sq_item = 44, t_object_ex = 16,
              file_input = 257;
constexpr unsigned tpflags_basetype = 1u << 10;
}  // namespace stable

}  // namespace neuron::python
