// PythonObject, and the functions with which HOC and the simulator call Python:
// callbacks, locations given as Python objects, Vector.from_python, SectionList of Python sections,
// pickled jobs, the GUI in Python, the names and cells of Python's sections.
//
// What each of them does is the interface's (Python) job: it registers one C function per hook with
// nrn_python_hook_set (docs/capi.rst). This file adapts the core's calls to them: it keeps the
// PythonObject and HOC's context for the call, and raises the HOC error of a hook that failed,
// after the hook has returned, so nothing is thrown through a Python frame. It does not call
// CPython: a Python object is a void* here, which only the interface reads.
#include <../../nrnconf.h>

#include <algorithm>
#include <array>
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
#include "nrnoc2iv.h"
#include "nrnpy.h"
#include "nrnpy_hooks.h"
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
extern "C" NRN_EXPORT int nrnpy_set_gui_callback(void*);
extern void oc_restore_hoc_oop(Object**, Objectdata**, int*, Symlist**);
extern void oc_save_hoc_oop(Object**, Objectdata**, int*, Symlist**);
extern NRN_DLLSYM void (*nrnpy_tree_changed_p_)();

namespace neuron::python {

bool (*import_interface)();

namespace {

// clang-format off
#define NRNPY_HOOKS(X)                                                                        \
    X(component) X(component_assign) X(to_number) X(operator_) X(statement) X(release)        \
    X(hold) X(command) X(apply) X(call_string) X(call_number) X(praxis) X(bind_arguments)      \
    X(call_with_section) X(command_tool) X(variable_get) X(variable_set) X(variable_string)   \
    X(gui_call) X(location) X(is_segment) X(segment_new) X(sections) X(numbers)               \
    X(vector_to_python) X(vector_as_numpy) X(pickle) X(unpickle) X(call_pickled)              \
    X(savestate_store) X(savestate_restore) X(coreneuron_option) X(coreneuron_arg)            \
    X(rvp_callable) X(wait) X(collective) X(section_name) X(section_cell)   \
    X(section_cell_equals) X(launcher)
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

constexpr std::size_t error_size = 1024;
using Err = char*;

// The C signatures (docs/capi.rst). A hook returns 1 when it is done, 0 with a message when it
// failed; 2 and 3 say more where noted. A void* is a Python object: what a PythonObject holds,
// borrowed, or null for HOC's `new PythonObject()`, which stands for __main__.
using Component = int (*)(Object*, void*, const char*, int, int, Err, std::size_t);
using ComponentAssign = int (*)(Object*, void*, Err, std::size_t);
using ToNumber = int (*)(Object*, double*);
using Operator = int (*)(void*, const char*, Object*, int, double, Err, std::size_t);
using Statement = int (*)(const char*, int*, Err, std::size_t);
using Release = void (*)(void*);
using Hold = int (*)(void*, void**);
using Command = int (*)(void*, Err, std::size_t);
using Apply = int (*)(void*, Object*, int, int, Err, std::size_t);
using CallString = int (*)(void*, nrn_bytes*, Err, std::size_t);
using CallNumber = int (*)(void*, int, int, double*, Err, std::size_t);
using Praxis = int (*)(void*, Object*, double*, Err, std::size_t);
using BindArguments = int (*)(void*, int, Object**, Err, std::size_t);
using CallWithSection = int (*)(void*, Section*, Err, std::size_t);
using CommandTool = int (*)(void*, int, double, double, int, Err, std::size_t);
using VariableGet = int (*)(void*, double*, Err, std::size_t);
using VariableSet = int (*)(void*, double, Err, std::size_t);
using VariableString = int (*)(void*, nrn_bytes*, Err, std::size_t);
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
using Location = int (*)(void*, int, Section**, double*, Err, std::size_t);
using IsSegment = int (*)(void*);
using SegmentNew = int (*)(Section*, double, Object**, Err, std::size_t);
using Sections = int (*)(void*, nrn_bytes*, Err, std::size_t);
using Numbers = int (*)(const char*, void*, nrn_bytes*, Err, std::size_t);
using VectorToPython =
    int (*)(double*, std::size_t, const char*, void*, Object**, Err, std::size_t);
using VectorAsNumpy = int (*)(double*, std::size_t, Object**, Err, std::size_t);
using Pickle = int (*)(void*, nrn_bytes*, Err, std::size_t);
using Unpickle = int (*)(const char*, std::size_t, Object**, Err, std::size_t);
using CallPickled = int (*)(const char*, std::size_t, int, nrn_bytes*, Err, std::size_t);
using SavestateStore = int (*)(nrn_bytes*, Err, std::size_t);
using SavestateRestore = int (*)(const char*, std::size_t, Err, std::size_t);
using CoreneuronOption = int (*)(int, int*);
using CoreneuronArg = int (*)(double, nrn_bytes*);
using RvpCallable = int (*)(void*, Object**, Err, std::size_t);
using Wait = void (*)(void (*)(void*), void*);
using Collective = int (*)(int, int, int, Object**, Err, std::size_t);
using SectionName = int (*)(void*, nrn_bytes*);
using SectionCell = int (*)(void*, Object**);
using SectionCellEquals = int (*)(void*, Object*);
using Launcher = int (*)(const char*, const char*);

/// Is the hook registered? Does not start Python.
bool has(Hook hook) {
    return hooks[static_cast<int>(hook)];
}

template <typename F>
F get(Hook hook) {
    return reinterpret_cast<F>(hooks[static_cast<int>(hook)]);
}

/// Start Python if need be, and import neuron, which registers the hooks; once.
void import_neuron() {
    if (!start()) {
        hoc_execerror("Python not available", nullptr);
    }
    static bool tried{};
    if (!import_interface || std::exchange(tried, true)) {
        return;
    }
    // Importing it reads HOC files, and HOC is in a line.
    std::string const line{hoc_cbufstr->buf};
    auto const position = hoc_ctp - hoc_cbufstr->buf;
    bool const imported = import_interface();
    std::strcpy(hoc_cbufstr->buf, line.c_str());
    hoc_ctp = hoc_cbufstr->buf + position;
    if (!imported) {
        hoc_execerror("could not import neuron", nullptr);
    }
}

/// The hook, which importing neuron registers; a HOC error if there is none.
template <typename F>
F need(Hook hook) {
    if (!has(hook)) {
        import_neuron();
    }
    if (!has(hook)) {
        hoc_execerror("The neuron package did not register its Python hooks", nullptr);
    }
    return get<F>(hook);
}

/// Python code can drop the last other reference to an object.
class Held {
    Object* o_;

  public:
    explicit Held(Object* o)
        : o_{o} {
        if (o_) {
            hoc_obj_ref(o_);
        }
    }
    Held(const Held&) = delete;
    ~Held() {
        if (o_) {
            hoc_obj_unref(o_);
        }
    }
};

/// A call of a hook: the hook is there, the object is held, HOC is at its top level. check() raises
/// the HOC error of a hook that failed.
class Call {
    Hook hook_;
    Held held_;
    Top top_;

  public:
    char err[error_size];
    explicit Call(Hook hook, Object* o = nullptr)
        : hook_{(need<Function>(hook), hook)}
        , held_{o} {
        err[0] = '\0';
    }
    template <typename F>
    F get() const {
        return python::get<F>(hook_);
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

std::string text(nrn_bytes const& bytes) {
    return {bytes.data.begin(), std::find(bytes.data.begin(), bytes.data.end(), '\0')};
}

/// A HOC string of what a hook returned. The slots outlive the expression that uses it, as
/// hoc_temp_charptr's do.
char** temp_string(nrn_bytes const& bytes) {
    static std::array<std::string, 128> slots;
    static std::size_t next{};
    auto& slot = slots[next++ % slots.size()];
    slot = text(bytes);
    char** const string = hoc_temp_charptr();
    *string = slot.data();
    return string;
}

// ---- PythonObject

void* construct(Object*) {
    return nullptr;
}

void destruct(void* p) {
    if (p && has(Hook::release)) {
        Top const top{};
        get<Release>(Hook::release)(p);
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

// Per event (NetCon.record, CVode.event, FInitializeHandler, ...). 2: the failure was only printed.
int hoccommand_exec(Object* o) {
    require_python_object(o, "not a Python callback");
    Call call{Hook::command, o};
    return call.check(call.get<Command>()(payload(o), call.err, sizeof call.err)) == 1;
}

// f(x) for each element from start to end.
void vec_apply(IvocVect* vec, Object* f, int start, int end) {
    require_python_object(f, "apply: first argument must be a HOC string or a Python callable");
    Call call{Hook::apply, f};
    call.check(call.get<Apply>()(payload(f), vec->obj_, start, end, call.err, sizeof call.err));
}

int hoccommand_exec_strret(Object* o, char* buf, int size) {
    require_python_object(o, "not a Python callback");
    Call call{Hook::call_string, o};
    nrn_bytes out;
    // 2: the failure was only printed.
    if (call.check(call.get<CallString>()(payload(o), &out, call.err, sizeof call.err)) == 2) {
        return 0;
    }
    std::strncpy(buf, text(out).c_str(), size);
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
    nrn_bytes out;
    call.check(call.get<VariableString>()(payload(o), &out, call.err, sizeof call.err));
    std::string const string(out.data.begin(), out.data.end());
    if (*cpp && string == *cpp) {
        return 0;
    }
    delete[] * cpp;
    *cpp = new char[string.size() + 1];
    std::strcpy(*cpp, string.c_str());
    return 1;
}

// ---- the GUI in Python: the interface's gui callback (name, the object of a method, HOC's object,
// arguments...)

/// Push the arguments of the HOC call in progress; a pointer as a Pointer to it (a bit set in
/// pointers).
void push_argument(int i, int bit, int& pointers) {
    if (hoc_is_object_arg(i)) {
        hoc_push_object(*hoc_objgetarg(i));
    } else if (hoc_is_pdouble_arg(i)) {
        static Symbol* const pointer = hoc_table_lookup("Pointer", hoc_built_in_symlist);
        Object* const o = hoc_new_object(pointer, new OcPointer("", hoc_hgetarg<double>(i)));
        hoc_push_object(o);
        pointers |= bit < 31 ? 1 << bit : 0;
    } else if (hoc_is_str_arg(i)) {
        hoc_pushstr(hoc_pgargstr(i));
    } else {
        hoc_pushx(*getarg(i));
    }
}

int push_arguments(int& pointers) {
    int n{};
    for (; ifarg(n + 1); ++n) {
        push_argument(n + 1, n, pointers);
    }
    return n;
}

/// 2 if the interface has no GUI callback.
int call_gui(const char* name, Object* obj, bool references, Object** result, nrn_bytes* string) {
    if (!has(Hook::gui_call)) {
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
    nrn_bytes out;
    if (call_gui(name, obj, references, nullptr, &out) == 2) {
        return nullptr;
    }
    return temp_string(out);
}

double object_to_double(Object* o) {
    if (!has(Hook::to_number)) {
        return 0.;
    }
    Held const held{o};
    double x{-1.};  // as -1 if it is no number, which the caller cannot tell from
                    // a value
    get<ToNumber>(Hook::to_number)(o, &x);
    return x;
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
    if (!is_python_object(o) || !payload(o) || !has(Hook::is_segment)) {
        return 0;
    }
    return get<IsSegment>(Hook::is_segment)(payload(o));
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

// The names and cells of the sections Python made (nrn_section_new_owned): the owner is the Python
// Section. NEURON asks for a name once, and keeps it.
char* section_name(Section* sec) {
    static std::string name;
    nrn_bytes out;
    if (has(Hook::section_name) &&
        get<SectionName>(Hook::section_name)(nrn_section_owner_get(sec), &out)) {
        name = text(out);
    } else {
        name = fmt::format("__nrnsec_{}", fmt::ptr(sec));
    }
    return name.data();
}

// NEURON asks about any section; only those the interface made have an owner.
Object* section_cell(Section* sec) {
    Object* cell{};
    if (void* const owner = nrn_section_owner_get(sec); owner && has(Hook::section_cell)) {
        get<SectionCell>(Hook::section_cell)(owner, &cell);
    }
    return cell;
}

int section_cell_equals(Section* sec, Object* o) {
    void* const owner = nrn_section_owner_get(sec);
    return owner && has(Hook::section_cell_equals) &&
           get<SectionCellEquals>(Hook::section_cell_equals)(owner, o);
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

/// type is 1 alltoall, 2 allgather, 3 gather, 4 broadcast, 5 scatter. For 3, 4 and 5 size is the
/// root rank. The interface pickles and sends with HOC's pc.alltoall.
Object* collective(int size, int type) {
    if (type == 1 || type == 5) {
        hoc_objgetarg(1);  // a HOC error if it is not an object, as NEURON's
    }
    Call call{Hook::collective};
    int pointer{};
    push_argument(1, 0, pointer);
    Object* result{};
    if (call.check(
            call.get<Collective>()(type, size, pointer, &result, call.err, sizeof call.err)) == 3) {
        throw std::runtime_error("Python exception");
    }
    return temp(result);
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
    if (!has(Hook::savestate_store)) {
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
    if (has(Hook::savestate_restore)) {
        Call call{Hook::savestate_restore};
        status = call.check(call.get<SavestateRestore>()(data, size, call.err, sizeof call.err));
    }
    if (status == 2 && size) {
        hoc_execerror("SaveState:", "Missing data restore function.");
    }
}

// CoreNEURON's options are read while the simulator is set up: an error is printed, not raised.
int coreneuron_option(int which) {
    if (!has(Hook::coreneuron_option)) {
        return 0;
    }
    Top const top{};
    int value{};
    get<CoreneuronOption>(Hook::coreneuron_option)(which, &value);
    return value;
}

char* coreneuron_arg(double tstop) {
    if (!has(Hook::coreneuron_arg)) {
        return nullptr;
    }
    Top const top{};
    nrn_bytes out;
    if (!get<CoreneuronArg>(Hook::coreneuron_arg)(tstop, &out)) {
        return nullptr;
    }
    return strdup(text(out).c_str());
}

// What to plot for the Python object a RangeVarPlot was given: an rxd species becomes a function.
Object* rvp_callable(Object* o) {
    if (!is_python_object(o) || !has(Hook::rvp_callable)) {
        return o;
    }
    Call call{Hook::rvp_callable, o};
    Object* result{};
    if (call.check(call.get<RvpCallable>()(payload(o), &result, call.err, sizeof call.err)) == 2) {
        return o;
    }
    return temp(result);
}

// Windows: wait for the GUI thread, which calls Python. The interface calls wait without the GIL
// (and without its HOC lock).
void without_gil(void (*wait)(void*), void* arg) {
    if (has(Hook::wait)) {
        get<Wait>(Hook::wait)(wait, arg);
    } else {
        wait(arg);
    }
}

// PlotShape's Python variable: a reference to what the PythonObject holds.
void* plotshape_object(Object* o) {
    void* held{};
    if (is_python_object(o) && has(Hook::hold)) {
        get<Hold>(Hook::hold)(payload(o), &held);
    }
    return held;
}

// nrnpython(): with no interface (HOC's own nrnpython("import webbrowser") must not import neuron)
// the launcher's statement hook runs it, and an error is printed as python prints it.
void nrnpython_statement() {
    int ok{};
    if (start()) {
        Call call{Hook::statement};
        call.check(call.get<Statement>()(gargstr(1), &ok, call.err, sizeof call.err));
    }
    hoc_retpushx(ok);
}
}  // namespace

bool is_python_object(Object* o) {
    return o && o->ctemplate->sym == nrnpy_pyobj_sym_;
}

void* payload(Object* o) {
    return o->u.this_pointer;
}

void require_python_object(Object* o, const char* message) {
    if (!is_python_object(o)) {
        hoc_execerror(message, nullptr);
    }
    if (!start()) {
        hoc_execerror("Python not available", nullptr);
    }
}

Top::Top()
    : object_{hoc_thisobject} {
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

void clear_hooks() {
    std::fill(std::begin(hooks), std::end(hooks), nullptr);
}

bool launcher_hook(const char* what, const char* arg) {
    return has(Hook::launcher) && get<Launcher>(Hook::launcher)(what, arg);
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
    methods.object_to_double = object_to_double;
    methods.hoccommand_exec_strret = hoccommand_exec_strret;
    methods.hoc_nrnpython = nrnpython_statement;
    methods.mpi_alltoall_type = collective;
    methods.opaque_obj2pyobj = opaque_object;
    methods.pickle2po = pickle2po;
    methods.po2pickle = po2pickle;
    methods.praxis_efun = praxis_efun;
    methods.pysame = same_object;
    methods.without_gil = without_gil;
    nrnpy_call_obj_method = [](Object* o, const char* method, Object* other) {
        return call_method(o, method, other, false, 0.);
    };
    nrnpy_call_obj_method_double = [](Object* o, const char* method, double x) {
        return call_method(o, method, nullptr, true, x);
    };
    nrnpy_decref = destruct;
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
    nrnpy_hoccommand_exec = methods.hoccommand_exec = hoccommand_exec;
    nrnpy_vec_apply_p_ = vec_apply;
    nrnpy_tree_changed_p_ = tree_changed;
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
            // Python's sections are owned by their objects, which HOC asks for names and cells.
            static bool owners{};
            if (hook && !owners) {
                owners = nrn_section_owner_hooks_set(
                    section_name, section_cell, section_cell_equals, nullptr, 0);
            }
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

Object* nrn_python_object_new(void* value) {
    Object* const o = hoc_new_object(nrnpy_pyobj_sym_, value);
    hoc_obj_ref(o);
    return o;
}

int nrn_python_object_get(Object* obj, void** value) {
    if (!is_python_object(obj)) {
        return 0;
    }
    *value = payload(obj);
    return 1;
}

// For NEURON's GUI in Python (neuron/__init__.py), with the GIL. 1: the interface has not
// registered its hooks.
int nrnpy_set_gui_callback(void* callback) {
    if (!has(Hook::gui_call) || !has(Hook::hold)) {
        return callback ? 1 : 0;
    }
    // A PythonObject of its own: the hook keeps what it needs.
    void* held{};
    Object* const o = callback && get<Hold>(Hook::hold)(callback, &held)
                          ? nrn_python_object_new(held)
                          : nullptr;
    char err[error_size];
    Object* result{};
    int const status = get<GuiCall>(
        Hook::gui_call)(nullptr, o, nullptr, 0, 0, 0, 0, &result, nullptr, err, sizeof err);
    if (o) {
        hoc_obj_unref(o);
    }
    return status == 1 ? 0 : 2;
}
