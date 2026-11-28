// PythonObject, and the functions with which HOC and the simulator call Python: callbacks,
// locations given as Python objects, Vector.from_python, SectionList of Python sections, pickled
// jobs, the GUI in Python.
//
// What each of them does is the interface's (Python) job: it registers one C function per hook with
// nrn_python_hook_set (docs/capi.rst). This file adapts the core's calls to them: it keeps the
// PythonObject and HOC's context for the call, and raises the HOC error of a hook that failed,
// after the hook has returned, so nothing is thrown through a Python frame.
#include <../../nrnconf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "classreg.h"
#include "hocstr.h"
#include "ivocvect.h"
#include "neuronapi.h"
#include "nrnmpi.h"
#include "nrnoc2iv.h"
#include "nrnpy_capi.h"
#include "nrnpy_hooks.h"
#include "nrnpy_types.h"
#include "oc_ansi.h"
#include "ocjump.h"
#include "ocpointer.h"
#include "parse.hpp"
#include "seclist.h"
#include "section.h"
#include <InterViews/resource.h>

/// What a hook returns as bytes, in memory NEURON owns.
struct nrn_bytes {
    std::vector<char> data;
};

extern Object** hoc_temp_objptr(Object*);
extern "C" NRN_EXPORT int nrnpy_set_gui_callback(PyObject*);
extern void oc_restore_hoc_oop(Object**, Objectdata**, int*, Symlist**);
extern void oc_save_hoc_oop(Object**, Objectdata**, int*, Symlist**);

namespace neuron::python {
namespace {

// clang-format off
#define NRNPY_HOOKS(X)                                                                        \
    X(component) X(component_assign) X(to_python) X(to_number) X(operator_) X(statement)      \
    X(callback_failed) X(call_string) X(call_number) X(praxis) X(bind_arguments)              \
    X(call_with_section) X(command_tool) X(variable_get) X(variable_set) X(variable_string)   \
    X(gui_call) X(location) X(is_segment) X(segment_new) X(sections) X(numbers)               \
    X(vector_to_python) X(vector_as_numpy) X(pickle) X(unpickle) X(call_pickled)              \
    X(savestate_store) X(savestate_restore) X(coreneuron_option) X(coreneuron_arg)            \
    X(rvp_callable) X(wait_begin) X(wait_end)
// clang-format on
enum class Hook {
#define X(name) name,
    NRNPY_HOOKS(X)
#undef X
        count
};
// The names as the interface gives them: operator_ is "operator".
std::string hook_name(Hook hook) {
    static char const* const names[]{
#define X(name) #name,
        NRNPY_HOOKS(X)
#undef X
    };
    std::string name = names[static_cast<int>(hook)];
    if (name.back() == '_') {
        name.pop_back();
    }
    return name;
}

using Function = void (*)();
Function hooks[static_cast<int>(Hook::count)];
bool hooks_registered{};

constexpr std::size_t error_size = 1024;
using Err = char*;

// The C signatures (docs/capi.rst). A hook returns 1 when it is done, 0 with a message when it
// failed; 2 and 3 say more where noted.
using Component = int (*)(Object*, void*, const char*, int, int, Err, std::size_t);
using ComponentAssign = int (*)(Object*, void*, Err, std::size_t);
using ToPython = int (*)(Object*, PyObject**, Err, std::size_t);
using ToNumber = int (*)(PyObject*, double*);
using Operator = int (*)(PyObject*, const char*, Object*, int, double, Err, std::size_t);
using Statement = int (*)(const char*, int*, Err, std::size_t);
using CallbackFailed = int (*)(int, PyObject*, Err, std::size_t);
using CallString = int (*)(PyObject*, nrn_bytes*, Err, std::size_t);
using CallNumber = int (*)(PyObject*, int, int, double*, Err, std::size_t);
using Praxis = int (*)(PyObject*, Object*, double*, Err, std::size_t);
using BindArguments = int (*)(PyObject*, int, Object**, Err, std::size_t);
using CallWithSection = int (*)(PyObject*, Section*, Err, std::size_t);
using CommandTool = int (*)(PyObject*, int, double, double, int, Err, std::size_t);
using VariableGet = int (*)(PyObject*, double*, Err, std::size_t);
using VariableSet = int (*)(PyObject*, double, Err, std::size_t);
using VariableString = int (*)(PyObject*, nrn_bytes*, Err, std::size_t);
using GuiCall = int (*)(const char*,
                        Object*,
                        Object*,
                        int,
                        int,
                        int,
                        int,
                        Object**,
                        nrn_bytes*,
                        Err,
                        std::size_t);
using Location = int (*)(PyObject*, int, Section**, double*, Err, std::size_t);
using IsSegment = int (*)(PyObject*);
using SegmentNew = int (*)(Section*, double, Object**, Err, std::size_t);
using Sections = int (*)(PyObject*, nrn_bytes*, Err, std::size_t);
using Numbers = int (*)(const char*, PyObject*, nrn_bytes*, Err, std::size_t);
using VectorToPython =
    int (*)(double*, std::size_t, const char*, PyObject*, Object**, Err, std::size_t);
using VectorAsNumpy = int (*)(double*, std::size_t, Object**, Err, std::size_t);
using Pickle = int (*)(PyObject*, nrn_bytes*, Err, std::size_t);
using Unpickle = int (*)(const char*, std::size_t, Object**, Err, std::size_t);
using CallPickled = int (*)(const char*, std::size_t, int, nrn_bytes*, Err, std::size_t);
using SavestateStore = int (*)(nrn_bytes*, Err, std::size_t);
using SavestateRestore = int (*)(const char*, std::size_t, Err, std::size_t);
using CoreneuronOption = int (*)(int, int*);
using CoreneuronArg = int (*)(double, nrn_bytes*);
using RvpCallable = int (*)(PyObject*, Object**, Err, std::size_t);
using Wait = void (*)();

/// Is the hook registered? Does not start Python.
bool has(Hook hook) {
    return hooks[static_cast<int>(hook)];
}

/// Importing neuron registers the hooks; once. The GIL is held.
void import_neuron() {
    static bool tried{};
    if (std::exchange(tried, true) || hooks_registered) {
        return;
    }
    // Importing it reads HOC files, and HOC is in a line.
    std::string const line{hoc_cbufstr->buf};
    auto const position = hoc_ctp - hoc_cbufstr->buf;
    Ref const neuron{py.PyImport_ImportModule("neuron")};
    std::strcpy(hoc_cbufstr->buf, line.c_str());
    hoc_ctp = hoc_cbufstr->buf + position;
    if (!neuron) {
        hoc_error_from_python("could not import neuron");
    }
}

/// The hook, which importing neuron registers; a HOC error if there is none. The GIL is held.
template <typename F>
F need(Hook hook) {
    if (!has(hook)) {
        import_neuron();
    }
    if (!has(hook)) {
        hoc_execerror("The neuron package did not register its Python hooks", nullptr);
    }
    return reinterpret_cast<F>(hooks[static_cast<int>(hook)]);
}

/// Python code can drop the last other reference to an object.
class Hold {
    Object* o_;

  public:
    explicit Hold(Object* o)
        : o_{o} {
        if (o_) {
            hoc_obj_ref(o_);
        }
    }
    Hold(const Hold&) = delete;
    ~Hold() {
        if (o_) {
            hoc_obj_unref(o_);
        }
    }
};

/// A call of a hook: Python runs, the object is held, HOC is at its top level. check() raises the
/// HOC error of a hook that failed.
class Call {
    Hook hook_;
    Hold hold_;
    Top top_;

  public:
    char err[error_size];
    explicit Call(Hook hook, Object* o = nullptr)
        : hook_{(require(), hook)}
        , hold_{o} {
        err[0] = '\0';
    }
    template <typename F>
    F get() const {
        return need<F>(hook_);
    }
    /// Raise the hook's message if it failed (0).
    int check(int status) {
        if (status == 0) {
            if (!err[0]) {
                std::snprintf(err, sizeof err, "Python hook %s failed", hook_name(hook_).c_str());
            }
            err[sizeof err - 1] = '\0';
            hoc_execerror(err, nullptr);
        }
        return status;
    }
};

/// A HOC object from the hook, with its reference given up for hoc_temp_objptr.
Object* temp(Object* o) {
    if (o) {
        --o->refcount;
    }
    return o;
}

/// A PythonObject with one reference.
Object* new_python_object(PyObject* p) {
    py.Py_IncRef(p);
    Object* const o = hoc_new_object(nrnpy_pyobj_sym_, p);
    hoc_obj_ref(o);
    return o;
}

/// The HOC object for a Python value, with a reference for the caller; null for None.
Object* to_hoc(PyObject* p) {
    if (p == py._Py_NoneStruct) {
        return nullptr;
    }
    if (is_instance(p, types.object)) {
        auto* const o = hoc_object(p);
        if (o) {
            hoc_obj_ref(o);
        }
        return o;
    }
    return new_python_object(p);
}

/// The Python value of a HOC object, a new reference. The GIL is held.
Ref to_python(Object* o) {
    if (!o) {
        return Ref::borrow(py._Py_NoneStruct);
    }
    if (is_python_object(o)) {
        return Ref::borrow(payload(o));
    }
    Call call{Hook::to_python};
    PyObject* value{};
    call.check(call.get<ToPython>()(o, &value, call.err, sizeof call.err));
    return Ref{value};
}

/// A HOC string of what a hook returned. The slots outlive the expression that uses it, as
/// hoc_temp_charptr's do.
char** temp_string(nrn_bytes const& bytes) {
    static std::array<std::string, 128> slots;
    static std::size_t next{};
    auto& slot = slots[next++ % slots.size()];
    slot.assign(bytes.data.begin(), std::find(bytes.data.begin(), bytes.data.end(), '\0'));
    char** const string = hoc_temp_charptr();
    *string = slot.data();
    return string;
}

// ---- PythonObject

void* construct(Object*) {
    return nullptr;
}

void destruct(void* p) {
    if (p && start()) {
        Top const top{};
        py.Py_DecRef(static_cast<PyObject*>(p));
    }
}

const char* component(Object* o, Symbol* sym, int nindex, int isfunc) {
    Call call{Hook::component, o};
    // 3: an item could not be got, and the message names the object.
    if (call.check(call.get<Component>()(
            o, o->u.this_pointer, sym->name, nindex, isfunc, call.err, sizeof call.err)) == 3) {
        hoc_execerror((std::string{call.err} + " " + hoc_object_name(o)).c_str(), nullptr);
    }
    return nullptr;
}

const char* assign_component(Object* o) {
    Call call{Hook::component_assign, o};
    call.check(call.get<ComponentAssign>()(o, o->u.this_pointer, call.err, sizeof call.err));
    return nullptr;
}

// ---- callbacks

int hoccommand_exec_strret(Object* o, char* buf, int size) {
    require_python_object(o, "not a Python callback");
    Call call{Hook::call_string, o};
    nrn_bytes text;
    // 2: the failure was only printed.
    if (call.check(call.get<CallString>()(payload(o), &text, call.err, sizeof call.err)) == 2) {
        return 0;
    }
    std::strncpy(buf, std::string(text.data.begin(), text.data.end()).c_str(), size);
    buf[size - 1] = '\0';
    return 1;
}

// err is 0 if a failure is only to be reported in it.
double call_func(Object* o, int narg, int* err) {
    require_python_object(o, "not a Python callback");
    Call call{Hook::call_number, o};
    double x{};
    int const report = !err || *err;
    int const status = call.check(
        call.get<CallNumber>()(payload(o), narg, report, &x, call.err, sizeof call.err));
    if (err) {
        *err = status == 2;
    }
    return status == 2 ? 0. : x;
}

double praxis_efun(Object* f, Object* vec) {
    require_python_object(f, "not a Python callback");
    Call call{Hook::praxis, f};
    double x{};
    // 3: float() of the result failed, which NEURON's Python module did not make a HOC error.
    if (call.check(call.get<Praxis>()(payload(f), vec, &x, call.err, sizeof call.err)) == 3) {
        throw std::runtime_error("Python exception");
    }
    return x;
}

Object* callable_with_args(Object* o, int narg) {
    require_python_object(o, "not a Python callback");
    Call call{Hook::bind_arguments, o};
    Object* result{};
    call.check(call.get<BindArguments>()(payload(o), narg, &result, call.err, sizeof call.err));
    return result;
}

void call_with_section(Object* o, Section* sec) {
    require_python_object(o, "not a Python callback");
    Call call{Hook::call_with_section, o};
    call.check(call.get<CallWithSection>()(payload(o), sec, call.err, sizeof call.err));
}

void command_tool(Object* o, int type, double x, double y, int key) {
    require_python_object(o, "not a Python callback");
    Call call{Hook::command_tool, o};
    call.check(call.get<CommandTool>()(payload(o), type, x, y, key, call.err, sizeof call.err));
}

// HOC's operators try a Python object's method first. 0 leaves the operator to HOC.
int call_method(Object* o, const char* method, Object* other, bool number, double x) {
    if (!is_python_object(o) || !start()) {
        return 0;
    }
    Call call{Hook::operator_, o};
    return call.check(call.get<Operator>()(
               payload(o), method, other, number, x, call.err, sizeof call.err)) == 1;
}

// ---- GUI variables: a (target, key) tuple

double gui_get(Object* o) {
    require_python_object(o, "not a Python variable");
    Call call{Hook::variable_get, o};
    double x{};
    call.check(call.get<VariableGet>()(payload(o), &x, call.err, sizeof call.err));
    return x;
}

void gui_set(Object* o, double x) {
    require_python_object(o, "not a Python variable");
    Call call{Hook::variable_set, o};
    call.check(call.get<VariableSet>()(payload(o), x, call.err, sizeof call.err));
}

// Returns whether the string changed.
int gui_get_str(Object* o, char** cpp) {
    require_python_object(o, "not a Python variable");
    Call call{Hook::variable_string, o};
    nrn_bytes text;
    call.check(call.get<VariableString>()(payload(o), &text, call.err, sizeof call.err));
    std::string const string(text.data.begin(), text.data.end());
    if (*cpp && string == *cpp) {
        return 0;
    }
    delete[] * cpp;
    *cpp = new char[string.size() + 1];
    std::strcpy(*cpp, string.c_str());
    return 1;
}

// ---- the GUI in Python: the interface's gui callback (name, the object of a
// method, HOC's object, arguments...)

/// Push the arguments of the HOC call in progress; a pointer as a Pointer to it (a bit set in
/// pointers).
int push_arguments(int& pointers) {
    int n{};
    for (; ifarg(n + 1); ++n) {
        int const i = n + 1;
        if (hoc_is_object_arg(i)) {
            hoc_push_object(*hoc_objgetarg(i));
        } else if (hoc_is_pdouble_arg(i)) {
            static Symbol* const pointer = hoc_table_lookup("Pointer", hoc_built_in_symlist);
            Object* const o = hoc_new_object(pointer, new OcPointer("", hoc_hgetarg<double>(i)));
            hoc_push_object(o);
            pointers |= n < 31 ? 1 << n : 0;
        } else if (hoc_is_str_arg(i)) {
            hoc_pushstr(hoc_pgargstr(i));
        } else {
            hoc_pushx(*getarg(i));
        }
    }
    return n;
}

/// 2 if the interface has no GUI callback.
int call_gui(const char* name, Object* obj, bool references, Object** result, nrn_bytes* string) {
    if (!has(Hook::gui_call) || !start()) {
        return 2;
    }
    // A destructor runs when the object is freed, maybe outside any HOC call: no arguments.
    bool const destructor = name[0] == '~';
    Object* const self = destructor ? nullptr : hoc_thisobject;
    Call call{Hook::gui_call};
    int pointers{};
    int const narg = destructor ? 0 : push_arguments(pointers);
    return call.check(call.get<GuiCall>()(name,
                                          obj,
                                          self,
                                          narg,
                                          pointers,
                                          references,
                                          string != nullptr,
                                          result,
                                          string,
                                          call.err,
                                          sizeof call.err));
}

Object** gui_helper3(const char* name, Object* obj, int references) {
    Object* result{};
    if (call_gui(name, obj, references, &result, nullptr) == 2) {
        return nullptr;
    }
    return hoc_temp_objptr(temp(result));
}

char** gui_helper3_str(const char* name, Object* obj, int references) {
    nrn_bytes text;
    if (call_gui(name, obj, references, nullptr, &text) == 2) {
        return nullptr;
    }
    return temp_string(text);
}

// ---- sections and locations

void locate(Object* o, int kind, const char* message, Section** sec, double* x) {
    require_python_object(o, message);
    Call call{Hook::location, o};
    call.check(call.get<Location>()(payload(o), kind, sec, x, call.err, sizeof call.err));
}

Section* o2sec(Object* o) {
    Section* sec{};
    double x{};
    locate(o, 0, "not a Python nrn.Section", &sec, &x);
    return sec;
}

void o2loc(Object* o, Section** sec, double* x) {
    locate(o, 1, "not a Python nrn.Segment", sec, x);
}

// Also an object with a segment attribute, such as an rxd node, or a list of one.
void o2loc2(Object* o, Section** sec, double* x) {
    locate(o, 2, "not a Python nrn.Segment, rxd.node, or other with a segment property", sec, x);
}

int is_segment(Object* o) {
    if (!is_python_object(o) || !has(Hook::is_segment) || !start()) {
        return 0;
    }
    Gil const gil{};
    return need<IsSegment>(Hook::is_segment)(payload(o));
}

Object* new_segment(Section* sec, double x) {
    Call call{Hook::segment_new};
    Object* result{};
    call.check(call.get<SegmentNew>()(sec, x, &result, call.err, sizeof call.err));
    return result;
}

void append_sections(void* list, Object* o) {
    require_python_object(o, "argument must be a Python iterable");
    Call call{Hook::sections, o};
    nrn_bytes out;
    // The sections before a failure are appended too.
    int const status = call.get<Sections>()(payload(o), &out, call.err, sizeof call.err);
    auto const n = out.data.size() / sizeof(Section*);
    for (std::size_t i = 0; i < n; ++i) {
        Section* sec;
        std::memcpy(&sec, out.data.data() + i * sizeof sec, sizeof sec);
        lvappendsec_and_ref(list, sec);
    }
    call.check(status);
}

// ---- Vector

IvocVect* vec_from_python(void* v) {
    auto* const vec = static_cast<IvocVect*>(v);
    Object* const o = *hoc_objgetarg(1);
    require_python_object(o, "the argument of Vector.from_python is not a PythonObject");
    Call call{Hook::numbers, o};
    nrn_bytes out;
    call.check(
        call.get<Numbers>()(hoc_object_name(o), payload(o), &out, call.err, sizeof call.err));
    vec->resize(out.data.size() / sizeof(double));
    std::memcpy(vec->data(), out.data.data(), vec->size() * sizeof(double));
    return vec;
}

Object** vec_to_python(void* v) {
    auto* const vec = static_cast<IvocVect*>(v);
    Object* const o = ifarg(1) ? *hoc_objgetarg(1) : nullptr;
    if (ifarg(1)) {
        require_python_object(o, "the argument of Vector.to_python is not a PythonObject");
    }
    Call call{Hook::vector_to_python, o};
    // Into the sequence o holds, or a new list.
    Object* result{};
    call.check(call.get<VectorToPython>()(vec->data(),
                                          vec->size(),
                                          o ? hoc_object_name(o) : nullptr,
                                          o ? payload(o) : nullptr,
                                          &result,
                                          call.err,
                                          sizeof call.err));
    return hoc_temp_objptr(o ? o : temp(result));
}

// The array shares the Vector's data.
Object** vec_as_numpy(int size, double* data) {
    Call call{Hook::vector_as_numpy};
    Object* result{};
    call.check(call.get<VectorAsNumpy>()(data, size, &result, call.err, sizeof call.err));
    return hoc_temp_objptr(temp(result));
}

// ---- ParallelContext

/// 3 and 4: NEURON's Python module threw a C++ exception there, which a method's caller makes the
/// HOC error "Class[i]::method: ...".
void check_pickling(Call& call, int status) {
    if (call.check(status) == 3) {
        throw std::runtime_error("Python exception");
    }
    if (status == 4) {
        throw std::bad_cast{};
    }
}

std::vector<char> po2pickle(Object* o) {
    if (!is_python_object(o)) {
        return {};
    }
    Call call{Hook::pickle, o};
    nrn_bytes out;
    check_pickling(call, call.get<Pickle>()(payload(o), &out, call.err, sizeof call.err));
    return std::move(out.data);
}

Object* pickle2po(const std::vector<char>& pickled) {
    Call call{Hook::unpickle};
    Object* result{};
    check_pickling(
        call,
        call.get<Unpickle>()(pickled.data(), pickled.size(), &result, call.err, sizeof call.err));
    return result;
}

std::vector<char> call_picklef(const std::vector<char>& pickled, int narg) {
    Call call{Hook::call_pickled};
    nrn_bytes out;
    check_pickling(call,
                   call.get<CallPickled>()(
                       pickled.data(), pickled.size(), narg, &out, call.err, sizeof call.err));
    return std::move(out.data);
}

#if NRNMPI
/// pickle.dumps or pickle.loads. Not released: Python may already be gone at exit.
PyObject* pickle_function(const char* name) {
    Ref const pickle{py.PyImport_ImportModule("pickle")};
    Ref function{pickle ? py.PyObject_GetAttrString(pickle.get(), name) : nullptr};
    if (!function) {
        hoc_error_from_python("Python's pickle is not available");
    }
    return function.release();
}

std::vector<char> dumps(PyObject* p) {
    static PyObject* function;
    if (!function) {
        function = pickle_function("dumps");
    }
    Ref const bytes = call(function, Ref::borrow(p));
    char* data{};
    std::ptrdiff_t size{};
    if (!bytes || py.PyBytes_AsStringAndSize(bytes.get(), &data, &size)) {
        hoc_error_from_python("Python exception");
    }
    return {data, data + size};
}

Ref loads(const char* data, std::size_t size) {
    static PyObject* function;
    if (!function) {
        function = pickle_function("loads");
    }
    Ref const bytes{py.PyBytes_FromStringAndSize(data, size)};
    Ref value = call(function, bytes);
    if (!value) {
        hoc_error_from_python("Python exception");
    }
    return value;
}

std::vector<int> displacements(const std::vector<int>& counts) {
    std::vector<int> d(counts.size() + 1);
    for (std::size_t i = 0; i < counts.size(); ++i) {
        d[i + 1] = d[i] + counts[i];
    }
    return d;
}

/// A list of the pickled values in data. A rank that sent nothing sent None.
Ref unpickle_list(const std::vector<char>& data,
                  const std::vector<int>& counts,
                  const std::vector<int>& displs) {
    Ref list{py.PyList_New(0)};
    for (std::size_t i = 0; i < counts.size(); ++i) {
        Ref const item = counts[i] ? loads(data.data() + displs[i], counts[i])
                                   : Ref::borrow(py._Py_NoneStruct);
        py.PyList_Append(list.get(), item.get());
    }
    return list;
}

Ref allgather(PyObject* source) {
    auto send = dumps(source);
    std::vector<int> counts(nrnmpi_numprocs);
    counts[nrnmpi_myid] = static_cast<int>(send.size());
    nrnmpi_int_allgather_inplace(counts.data(), 1);
    auto displs = displacements(counts);
    std::vector<char> data(displs.back());
    nrnmpi_char_allgatherv(send.data(), data.data(), counts.data(), displs.data());
    return unpickle_list(data, counts, displs);
}

Ref gather(PyObject* source, int root) {
    auto send = dumps(source);
    bool const is_root = nrnmpi_myid == root;
    int size = static_cast<int>(send.size());
    std::vector<int> counts(is_root ? nrnmpi_numprocs : 0);
    nrnmpi_int_gather(&size, counts.data(), 1, root);
    auto displs = displacements(counts);
    std::vector<char> data(displs.back());
    nrnmpi_char_gatherv(send.data(), size, data.data(), counts.data(), displs.data(), root);
    return is_root ? unpickle_list(data, counts, displs) : Ref::borrow(py._Py_NoneStruct);
}

Ref broadcast(PyObject* source, int root) {
    bool const is_root = nrnmpi_myid == root;
    auto data = is_root ? dumps(source) : std::vector<char>{};
    int size = static_cast<int>(data.size());
    nrnmpi_int_broadcast(&size, 1, root);
    data.resize(size);
    nrnmpi_char_broadcast(data.data(), size, root);
    return is_root ? Ref::borrow(source) : loads(data.data(), data.size());
}

/// The pickled items of a list, one per rank, None as nothing.
std::vector<char> pickle_items(PyObject* list, std::vector<int>& counts, bool keep) {
    std::vector<char> data;
    for (int i = 0; i < nrnmpi_numprocs; ++i) {
        auto* const item = py.PyList_GetItem(list, i);
        auto const pickled = item == py._Py_NoneStruct ? std::vector<char>{} : dumps(item);
        counts.push_back(static_cast<int>(pickled.size()));
        if (keep) {
            data.insert(data.end(), pickled.begin(), pickled.end());
        }
    }
    return data;
}

Ref scatter(PyObject* source, int root) {
    std::vector<int> counts;
    auto send = nrnmpi_myid == root ? pickle_items(source, counts, true) : std::vector<char>{};
    auto displs = displacements(counts);
    int size{};
    nrnmpi_int_scatter(counts.data(), &size, 1, root);
    std::vector<char> data(size + 1);  // not empty
    nrnmpi_char_scatterv(send.data(), counts.data(), displs.data(), data.data(), size, root);
    return size ? loads(data.data(), size) : Ref::borrow(py._Py_NoneStruct);
}

// A negative size asks only for the numbers of bytes to be sent and received.
Ref alltoall(PyObject* source, int size) {
    std::vector<int> send_counts;
    auto send = pickle_items(source, send_counts, size >= 0);
    std::vector<int> ones(nrnmpi_numprocs, 1);
    auto unit = displacements(ones);
    std::vector<int> counts(nrnmpi_numprocs);
    nrnmpi_int_alltoallv(
        send_counts.data(), ones.data(), unit.data(), counts.data(), ones.data(), unit.data());
    auto send_displs = displacements(send_counts);
    auto displs = displacements(counts);
    if (size < 0) {
        Ref tuple{py.PyTuple_New(2)};
        py.PyTuple_SetItem(tuple.get(), 0, py.PyLong_FromLongLong(send_displs.back()));
        py.PyTuple_SetItem(tuple.get(), 1, py.PyLong_FromLongLong(displs.back()));
        return tuple;
    }
    std::vector<char> data(displs.back() + 1);  // not empty
    nrnmpi_char_alltoallv(send.data(),
                          send_counts.data(),
                          send_displs.data(),
                          data.data(),
                          counts.data(),
                          displs.data());
    return unpickle_list(data, counts, displs);
}
#endif

/// The first argument of the HOC call in progress.
Ref first_argument() {
    if (hoc_is_object_arg(1)) {
        return to_python(*hoc_objgetarg(1));
    }
    if (hoc_is_str_arg(1)) {
        return Ref{py.PyUnicode_FromString(gargstr(1))};
    }
    return Ref{py.PyFloat_FromDouble(*getarg(1))};
}

/// type is 1 alltoall, 2 allgather, 3 gather, 4 broadcast, 5 scatter. For 3, 4 and 5 size is the
/// root rank.
Object* collective(int size, int type) {
    require();
    Top const top{};
    enum { ALLTOALL = 1, ALLGATHER, GATHER, BROADCAST, SCATTER };
    int const nhost = nrnmpi_numprocs;
    if (type == ALLTOALL || type == SCATTER) {
        hoc_objgetarg(1);  // a HOC error if it is not an object, as NEURON's
    }
    Ref const source = first_argument();
    if (type == ALLTOALL || (type == SCATTER && nrnmpi_myid == size)) {
        if (!is_instance(source.get(), py.PyList_Type)) {
            hoc_execerror("Argument must be a Python list", nullptr);
        }
        if (py.PyList_Size(source.get()) != nhost) {
            hoc_execerror(type == ALLTOALL ? "py_alltoall list size must be nhost"
                                           : "py_scatter list size must be nhost",
                          nullptr);
        }
    }
    if (nhost == 1) {
        switch (type) {
        case SCATTER:
            return temp(to_hoc(py.PyList_GetItem(source.get(), 0)));
        case ALLGATHER:
        case GATHER: {
            Ref const list{py.PyList_New(0)};
            py.PyList_Append(list.get(), source.get());
            return temp(to_hoc(list.get()));
        }
        default:
            return temp(to_hoc(source.get()));
        }
    }
#if NRNMPI
    if (type != ALLTOALL && type != ALLGATHER && (size < 0 || size >= nhost)) {
        hoc_execerror("root rank must be >= 0 and < nhost", nullptr);
    }
    Ref result;
    switch (type) {
    case ALLTOALL:
        result = alltoall(source.get(), size);
        break;
    case ALLGATHER:
        result = allgather(source.get());
        break;
    case GATHER:
        result = gather(source.get(), size);
        break;
    case BROADCAST:
        result = broadcast(source.get(), size);
        break;
    default:
        result = scatter(source.get(), size);
    }
    Object* const o = to_hoc(result.get());
    result = Ref{};  // before the reference is given up: it may hold the object's
                     // last other one
    return temp(o);
#else
    return nullptr;
#endif
}

// ---- other hooks

void* opaque_object(Object* o) {
    return is_python_object(o) ? o->u.this_pointer : nullptr;
}

int same_object(Object* a, Object* b) {
    return is_python_object(a) && is_python_object(b) && a->u.this_pointer == b->u.this_pointer;
}

// SaveState keeps the bytes the interface's store function returns, and gives them back to restore.
// 2 from a hook: the interface keeps no state.
void savestate_store(char** data, uint64_t* size) {
    *size = 0;
    if (!has(Hook::savestate_store) || !start()) {
        return;
    }
    Call call{Hook::savestate_store};
    nrn_bytes out;
    if (call.check(call.get<SavestateStore>()(&out, call.err, sizeof call.err)) == 2) {
        return;
    }
    delete[] * data;
    *data = new char[out.data.size()];
    std::memcpy(*data, out.data.data(), out.data.size());
    *size = out.data.size();
}

void savestate_restore(int64_t size, char* data) {
    int status = 2;
    if (has(Hook::savestate_restore) && start()) {
        Call call{Hook::savestate_restore};
        status = call.check(call.get<SavestateRestore>()(data, size, call.err, sizeof call.err));
    }
    if (status == 2 && size) {
        hoc_execerror("SaveState:", "Missing data restore function.");
    }
}

// CoreNEURON's options are read while the simulator is set up: an error is printed, not raised.
int coreneuron_option(int which) {
    if (!has(Hook::coreneuron_option) || !start()) {
        return 0;
    }
    Top const top{};
    int value{};
    need<CoreneuronOption>(Hook::coreneuron_option)(which, &value);
    return value;
}

char* coreneuron_arg(double tstop) {
    if (!has(Hook::coreneuron_arg) || !start()) {
        return nullptr;
    }
    Top const top{};
    nrn_bytes out;
    if (!need<CoreneuronArg>(Hook::coreneuron_arg)(tstop, &out)) {
        return nullptr;
    }
    return strdup(std::string(out.data.begin(), out.data.end()).c_str());
}

// What to plot for the Python object a RangeVarPlot was given: an rxd species becomes a function.
Object* rvp_callable(Object* o) {
    if (!is_python_object(o) || !has(Hook::rvp_callable) || !start()) {
        return o;
    }
    Call call{Hook::rvp_callable, o};
    Object* result{};
    if (call.check(call.get<RvpCallable>()(payload(o), &result, call.err, sizeof call.err)) == 2) {
        return o;
    }
    return temp(result);
}

// PlotShape's Python variable.
void* plotshape_object(Object* o) {
    if (!is_python_object(o) || !start()) {
        return nullptr;
    }
    Gil const gil{};
    return Ref::borrow(payload(o)).release();
}

void plotshape_release(void* p) {
    if (p && start()) {
        Top const top{};
        py.Py_DecRef(static_cast<PyObject*>(p));
    }
}

// Without the interface (HOC's own nrnpython("import webbrowser") must not import neuron), the
// statement runs here, and an error is printed as python prints it.
void nrnpython_statement() {
    int ok{};
    if (start() && has(Hook::statement)) {
        Call call{Hook::statement};
        call.check(call.get<Statement>()(gargstr(1), &ok, call.err, sizeof call.err));
    } else if (started()) {
        Top const top{};
        ok = static_cast<bool>(run(gargstr(1)));
        if (!ok) {
            py.PyErr_Print();
        }
    }
    hoc_retpushx(ok);
}

// Windows: this thread waits for the GUI's, which calls Python and HOC. The interface releases its
// lock as often as this thread holds it, and takes it as often again.
PyThreadState* save_thread() {
    if (!started()) {
        return nullptr;
    }
    if (has(Hook::wait_begin)) {
        Gil const gil{};
        need<Wait>(Hook::wait_begin)();
    }
    return py.PyEval_SaveThread();
}

void restore_thread(PyThreadState* state) {
    if (!state) {
        return;
    }
    py.PyEval_RestoreThread(state);
    if (has(Hook::wait_end)) {
        need<Wait>(Hook::wait_end)();
    }
}
}  // namespace

bool is_python_object(Object* o) {
    return o && o->ctemplate->sym == nrnpy_pyobj_sym_;
}

PyObject* payload(Object* o) {
    auto* const object = static_cast<PyObject*>(o->u.this_pointer);
    return object ? object : py.PyImport_AddModule("__main__");
}

void require_python_object(Object* o, const char* message) {
    if (!is_python_object(o)) {
        hoc_execerror(message, nullptr);
    }
    require();
}

Top::Top()
    : object_{hoc_thisobject} {
    model_changed();
    if (object_) {
        exceptions_ = std::uncaught_exceptions();
        oc_save_hoc_oop(&object_, &data_, &local_, &symbols_);
        hoc_thisobject = nullptr;
        hoc_objectdata = hoc_top_level_data;
        hoc_symlist = hoc_top_level_symlist;
    }
}

Top::~Top() {
    // A HOC error has put HOC where it must be.
    if (object_ && std::uncaught_exceptions() == exceptions_) {
        oc_restore_hoc_oop(&object_, &data_, &local_, &symbols_);
    }
}

void callback_failed(int kind) {
    PyObject *t{}, *v{}, *tb{};
    py.PyErr_Fetch(&t, &v, &tb);
    py.PyErr_NormalizeException(&t, &v, &tb);
    if (tb) {
        py.PyException_SetTraceback(v, tb);
    }
    Ref type{t}, value{v}, traceback{tb};
    char err[error_size]{};
    if (!has(Hook::callback_failed)) {
        py.PyErr_Restore(type.release(), value.release(), traceback.release());
        py.PyErr_Print();
        hoc_execerror("Python callback failed", nullptr);
    }
    if (need<CallbackFailed>(Hook::callback_failed)(kind, value.get(), err, sizeof err) == 1) {
        return;
    }
    hoc_execerror(err[0] ? err : "Python callback failed", nullptr);
}

void clear_hooks() {
    std::fill(std::begin(hooks), std::end(hooks), nullptr);
}

void hoc_error_from_python(const std::string& message) {
    py.PyErr_Print();
    hoc_execerror(message.c_str(), nullptr);
}

void require() {
    if (!start()) {
        hoc_execerror("Python not available", nullptr);
    }
}

void register_python_object() {
    class2oc("PythonObject", construct, destruct, nullptr, nullptr, nullptr);
    nrnpy_pyobj_sym_ = hoc_lookup("PythonObject");
    nrn_template_set_component_hooks(nrnpy_pyobj_sym_, component, assign_component, nullptr, 0);

    methods.callable_with_args = callable_with_args;
    methods.call_func = call_func;
    methods.call_picklef = call_picklef;
    methods.call_python_with_section = call_with_section;
    methods.cmdtool = command_tool;
    methods.guigetstr = gui_get_str;
    methods.guigetval = gui_get;
    methods.guisetval = gui_set;
    methods.gui_helper = [](const char* name, Object* obj) { return gui_helper3(name, obj, 0); };
    methods.gui_helper3 = gui_helper3;
    methods.gui_helper3_str = gui_helper3_str;
    methods.object_to_double = [](Object* o) {
        if (!started() || !has(Hook::to_number)) {
            return 0.;
        }
        Gil const gil{};
        Ref const value = to_python(o);
        double x{-1.};  // as -1 if it is no number, which the caller cannot tell
                        // from a value
        need<ToNumber>(Hook::to_number)(value.get(), &x);
        return x;
    };
    methods.hoccommand_exec_strret = hoccommand_exec_strret;
    methods.hoc_nrnpython = nrnpython_statement;
    methods.ho2po = [](Object* o) { return to_python(o).release(); };
    methods.mpi_alltoall_type = collective;
    methods.opaque_obj2pyobj = opaque_object;
    methods.pickle2po = pickle2po;
    methods.po2ho = to_hoc;
    methods.po2pickle = po2pickle;
    methods.praxis_efun = praxis_efun;
    methods.pysame = same_object;
    methods.restore_thread = restore_thread;
    methods.save_thread = save_thread;
    nrnpy_call_obj_method = [](Object* o, const char* method, Object* other) {
        return call_method(o, method, other, false, 0.);
    };
    nrnpy_call_obj_method_double = [](Object* o, const char* method, double x) {
        return call_method(o, method, nullptr, true, x);
    };
    nrnpy_decref = plotshape_release;
    nrnpy_get_pyobj = plotshape_object;
    nrnpy_nrncore_arg_p_ = coreneuron_arg;
    nrnpy_nrncore_enable_value_p_ = [] { return coreneuron_option(0); };
    nrnpy_nrncore_file_mode_value_p_ = [] { return coreneuron_option(1); };
    nrnpy_o2loc_p_ = o2loc;
    nrnpy_o2loc2_p_ = o2loc2;
    nrnpy_o2sec_p_ = o2sec;
    nrnpy_ob_is_seg = is_segment;
    nrnpy_restore_savestate = savestate_restore;
    nrnpy_rvp_rxd_to_callable = rvp_callable;
    nrnpy_sectionlist_helper_ = append_sections;
    nrnpy_seg_from_sec_x = new_segment;
    nrnpy_store_savestate = savestate_store;
    nrnpy_vec_as_numpy_helper_ = vec_as_numpy;
    nrnpy_vec_from_python_p_ = vec_from_python;
    nrnpy_vec_to_python_p_ = vec_to_python;
    fast::register_callbacks();
}

}  // namespace neuron::python

using namespace neuron::python;

int nrn_python_hook_set(const char* name,
                        void (*hook)(void),
                        char* error_msg,
                        size_t error_msg_size) {
    for (int i = 0; i < static_cast<int>(Hook::count); ++i) {
        if (name && hook_name(static_cast<Hook>(i)) == name) {
            hooks[i] = hook;
            hooks_registered = hooks_registered || hook;
            fast::forget_classes();
            return 0;
        }
    }
    if (error_msg && error_msg_size) {
        std::snprintf(error_msg,
                      error_msg_size,
                      "%s is not a Python hook",
                      name ? name : "a null name");
    }
    return 1;
}

void nrn_bytes_set(nrn_bytes* out, const char* data, size_t size) {
    out->data.assign(data, data + size);
}

Object* nrn_python_object_new(PyObject* value) {
    return start() ? new_python_object(value) : nullptr;
}

PyObject* nrn_python_object_get(Object* obj) {
    if (!start()) {
        return nullptr;
    }
    if (!obj) {
        return Ref::borrow(py._Py_NoneStruct).release();
    }
    return is_python_object(obj) ? Ref::borrow(payload(obj)).release() : nullptr;
}

PyObject* nrn_python_type(const char* name) {
    if (!start() || !make_types()) {
        return nullptr;
    }
    auto* const type = !name                                                ? py._Py_NoneStruct
                       : !std::strcmp(name, "object")                       ? types.object
                       : !std::strcmp(name, "section")                      ? types.section
                       : !std::strcmp(name, "segment")                      ? types.segment
                       : !std::strcmp(name, "mechanism") && types.mechanism ? types.mechanism
                       : !std::strcmp(name, "vector") && types.vector       ? types.vector
                                                                            : py._Py_NoneStruct;
    return Ref::borrow(type).release();
}

// Called from Python, with the GIL. 1: the interface has not registered its hooks.
int nrnpy_set_gui_callback(PyObject* callback) {
    if (!started()) {
        return 3;
    }
    bool const none = !callback || callback == py._Py_NoneStruct;
    if (!has(Hook::gui_call)) {
        return none ? 0 : 1;
    }
    Object* const o = none ? nullptr : new_python_object(callback);
    char err[error_size];
    Object* result{};
    int const status = reinterpret_cast<GuiCall>(hooks[static_cast<int>(Hook::gui_call)])(
        nullptr, o, nullptr, 0, 0, 0, 0, &result, nullptr, err, sizeof err);
    if (o) {
        hoc_obj_unref(o);  // the interface keeps its own
    }
    return status == 1 ? 0 : 2;
}
