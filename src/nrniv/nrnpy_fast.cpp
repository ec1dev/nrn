// The base types of the interface's wrappers, and the fast path of its hot accesses.
//
// hoc._Object and nrn._Section hold what a wrapper stands for (_obj, _sec: an int) and release it
// in tp_dealloc, which runs after a subclass's __del__ and after the collector: neither can skip
// it. The interface (Python) derives its classes from them and adds the other slots.
//
// The fast path decides only the common case: a scalar range variable of a segment or of a
// density mechanism at an interior x, a Vector element at an int index. The rules:
// - It reads only tables and slots that the interface owns and fills after its own slow path
//   resolved a name (nrn_python_fast_setup). It never resolves a name or caches anything itself.
// - The symbol tables are cleared here whenever KSChan renames or frees a symbol
//   (nrn_kschan_symbols_changed); no other code renames or frees a range variable's symbol.
// - Nothing is raised here. Anything else, and every failure, calls the interface's method
//   (_getattr, _setattr, _getitem, _setitem), which raises NEURON's error.
// - Only exact instances of the interface's Segment, Mechanism and Vector take it.
// - NRN_PYTHON_FAST_PATH=0 (exactly "0", as NEURON compares its other switches) leaves it out.
#include <../../nrnconf.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "neuronapi.h"
#include "nrnpy.h"
#include "nrnpy_capi.h"

extern std::size_t nrn_kschan_symbols_changed;

namespace neuron::python {
namespace {
using stable::Wrapper;

/// What the interface gives nrn_python_fast_setup (config): its classes, its model epoch (a list
/// of one, None once the model may have changed), section address -> nseg generation, the name
/// tables (Symbol addresses), 0 and AttributeError; and where the slots of a Segment and a
/// Mechanism are (offset).
enum { SegmentClass, MechanismClass, VectorClass, Epoch, Generations, RangeVars, Mechanisms };
enum { Zero = Mechanisms + 1, AttributeError };
enum { SegSec, SegX, SegCache, MechSeg, MechX, MechName, MechEpoch, MechNseg };
PyObject* config[9];
std::ptrdiff_t offset[8];
PyObject* types[5];  // object, section, segment, mechanism, vector

PyObject* type_of(PyObject* o) {
    return reinterpret_cast<Wrapper*>(o)->type;
}

PyObject*& at(PyObject* o, std::ptrdiff_t offset) {
    return *reinterpret_cast<PyObject**>(reinterpret_cast<char*>(o) + offset);
}

/// What the fast path leaves to the interface: its method (_getattr, ...).
PyObject* call(PyObject* self, const char* method, PyObject* a = nullptr, PyObject* b = nullptr) {
    return py.PyObject_CallMethod(self, method, a ? (b ? "OO" : "O") : nullptr, a, b);
}

int status(PyObject* result) {
    py.Py_DecRef(result);
    return result ? 0 : -1;
}

void* address(PyObject* wrapper) {
    PyObject* const a = reinterpret_cast<Wrapper*>(wrapper)->address;
    return a && type_of(a) == py.PyLong_Type ? py.PyLong_AsVoidPtr(a) : nullptr;
}

/// The section of a Section wrapper, or null.
Section* section_of(PyObject* wrapper) {
    bool const section = wrapper && py.PyType_IsSubtype(type_of(wrapper), types[1]);
    return section ? static_cast<Section*>(address(wrapper)) : nullptr;
}

// ---- the base types

template <void (*release)(PyObject*)>
void dealloc(PyObject* self) {
    auto* const w = reinterpret_cast<Wrapper*>(self);
    PyObject* const type = type_of(self);
    if (py.Py_IsInitialized()) {  // at exit Python is gone for HOC, and what w holds stays
        PyObject *t, *v, *tb;     // while HOC releases it, an exception being handled waits
        py.PyErr_Fetch(&t, &v, &tb);
        release(self);
        py.PyErr_Restore(t, v, tb);
    }
    py.Py_DecRef(w->address);
    py.Py_DecRef(w->pynative);
    reinterpret_cast<void (*)(void*)>(py.PyType_GetSlot(type, stable::tp_free))(self);
    py.Py_DecRef(type);
}

void release_object(PyObject* self) {
    if (auto* const o = static_cast<Object*>(address(self))) {
        nrn_object_unref(o);
    }
}

// A section that Python made goes with its object. A wrapper only refers to one that HOC made.
void release_section(PyObject* self) {
    if (auto* const sec = static_cast<Section*>(address(self))) {
        bool const python_made = reinterpret_cast<Wrapper*>(self)->pynative != py._Py_FalseStruct;
        python_made ? nrn_section_owned_release(sec) : nrn_section_unref(sec);
    }
}

// ---- Segment and Mechanism

/// An underscore name that is not _ref_*: one of the interface's slots or methods.
bool internal(PyObject* name) {
    char const* const s = py.PyUnicode_AsUTF8AndSize(name, nullptr);
    if (!s) {
        py.PyErr_Clear();
    }
    return s && s[0] == '_' && std::strncmp(s, "_ref_", 5) != 0;
}

/// An interior x from a float slot.
bool interior(PyObject* position, double& x) {
    return position && type_of(position) == py.PyFloat_Type &&
           (x = py.PyFloat_AsDouble(position)) > 0.0 && x < 1.0;
}

/// Is a wrapper's epoch, which the interface set when it checked the wrapper, the current one?
bool current(PyObject* epoch) {
    return epoch && epoch == py.PyList_GetItem(config[Epoch], 0) && epoch != py._Py_NoneStruct;
}

Section* segment_location(PyObject* segment, double& x) {
    Section* const sec = section_of(at(segment, offset[SegSec]));
    return sec && interior(at(segment, offset[SegX]), x) ? sec : nullptr;
}

/// Section and interior x of a Segment, or of a Mechanism checked at this epoch.
Section* location(PyObject* self, bool segment, double& x) {
    PyObject* const seg = segment ? self : at(self, offset[MechSeg]);
    bool const checked = segment ||
                         (seg && py.PyType_IsSubtype(type_of(seg), config[SegmentClass]) &&
                          current(at(self, offset[MechEpoch])));
    Section* const sec = checked ? segment_location(seg, x) : nullptr;
    return sec && (segment || interior(at(self, offset[MechX]), x)) ? sec : nullptr;
}

/// A Segment's cached Mechanism, when the interface's own test would accept it: its epoch and its
/// section's nseg generation are the current ones (Segment._getattr). A new reference.
PyObject* cached_mechanism(PyObject* self, PyObject* name) {
    PyObject* const cache = at(self, offset[SegCache]);
    PyObject* const mech = cache && type_of(cache) == py.PyDict_Type
                               ? py.PyDict_GetItem(cache, name)
                               : nullptr;
    PyObject* const section = at(self, offset[SegSec]);
    if (!mech || !py.PyType_IsSubtype(type_of(mech), config[MechanismClass]) ||
        !current(at(mech, offset[MechEpoch])) || !section_of(section)) {
        return nullptr;
    }
    // The interface took the generation from that dict, or 0 for a section that has none there.
    PyObject* const sec = reinterpret_cast<Wrapper*>(section)->address;
    PyObject* const generation = py.PyDict_GetItem(config[Generations], sec);
    if (at(mech, offset[MechNseg]) != (generation ? generation : config[Zero])) {
        return nullptr;
    }
    py.Py_IncRef(mech);
    return mech;
}

/// The Symbol of a scalar range variable of the segment, or of the mechanism, or null.
Symbol* lookup(PyObject* self, PyObject* name, bool segment) {
    static std::size_t seen{};
    if (seen != nrn_kschan_symbols_changed) {  // forget every name, KSChan renamed or freed one
        seen = nrn_kschan_symbols_changed;
        py.PyDict_Clear(config[RangeVars]);
        py.PyDict_Clear(config[Mechanisms]);
    }
    PyObject* names = config[RangeVars];
    if (!segment) {
        PyObject* const mech = at(self, offset[MechName]);
        names = mech && type_of(mech) == py.PyUnicode_Type
                    ? py.PyDict_GetItem(config[Mechanisms], mech)
                    : nullptr;
    }
    PyObject* const symbol = names ? py.PyDict_GetItem(names, name) : nullptr;
    return symbol ? static_cast<Symbol*>(py.PyLong_AsVoidPtr(symbol)) : nullptr;
}

/// tp_getattro of Segment and Mechanism. A Python subclass that defines __getattr__ reaches it
/// through CPython's hook, which then calls that __getattr__, as without the fast path.
template <bool segment>
PyObject* getattro(PyObject* self, PyObject* name) {
    bool const fast = type_of(self) == (segment ? config[SegmentClass] : config[MechanismClass]);
    if (fast && type_of(name) == py.PyUnicode_Type && !internal(name)) {
        if (Symbol* const sym = lookup(self, name, segment)) {
            double x;
            Section* const sec = location(self, segment, x);
            double const v = sec ? nrn_rangevar_get_or_nan(sym, sec, x) : NAN;
            return std::isnan(v) ? call(self, "_getattr", name) : py.PyFloat_FromDouble(v);
        }
        if (segment && py.PyDict_GetItem(config[Mechanisms], name)) {
            PyObject* const mech = cached_mechanism(self, name);
            return mech ? mech : call(self, "_getattr", name);
        }
    }
    PyObject* const found = py.PyObject_GenericGetAttr(self, name);
    if (found || !py.PyErr_ExceptionMatches(config[AttributeError]) ||
        (!fast && py.PyType_GetSlot(type_of(self), stable::tp_getattro) !=
                      reinterpret_cast<void*>(getattro<segment>))) {
        return found;  // the subclass's own __getattr__ comes next
    }
    py.PyErr_Clear();
    return call(self, "_getattr", name);
}

/// tp_setattro of Segment and Mechanism: an exact int or float to a range variable.
template <bool segment>
int setattro(PyObject* self, PyObject* name, PyObject* value) {
    if (!value || type_of(name) != py.PyUnicode_Type || internal(name)) {
        return py.PyObject_GenericSetAttr(self, name, value);
    }
    Symbol* const sym = type_of(self) == (segment ? config[SegmentClass] : config[MechanismClass])
                            ? lookup(self, name, segment)
                            : nullptr;
    double x;
    Section* const sec = sym && (type_of(value) == py.PyFloat_Type ||
                                 type_of(value) == py.PyLong_Type)
                             ? location(self, segment, x)
                             : nullptr;
    if (sec) {
        double const d = py.PyFloat_AsDouble(value);
        // diam and POINTERs are never in the tables; set_nothrow does the rest of HOC's assignment.
        if (!(d == -1.0 && py.PyErr_Occurred()) &&
            nrn_rangevar_set_nothrow(sym, sec, x, 0, d, nullptr, 0) == 0) {
            return 0;
        }
        py.PyErr_Clear();
    }
    return status(call(self, "_setattr", name, value));
}

// ---- Vector elements

/// The element of an exact Vector for an int key in range, or null.
double* element(PyObject* self, PyObject* key) {
    auto* const o = type_of(self) == config[VectorClass] && type_of(key) == py.PyLong_Type
                        ? static_cast<Object*>(address(self))
                        : nullptr;
    std::ptrdiff_t i = o ? py.PyLong_AsSsize_t(key) : -1;
    if (!o || (i == -1 && py.PyErr_Occurred())) {
        py.PyErr_Clear();
        return nullptr;
    }
    std::ptrdiff_t const size = nrn_vector_capacity(o);
    i += i < 0 ? size : 0;
    return i >= 0 && i < size ? nrn_vector_data(o) + i : nullptr;
}

PyObject* vector_getitem(PyObject* self, PyObject* key) {
    double const* const pd = element(self, key);
    return pd ? py.PyFloat_FromDouble(*pd) : call(self, "_getitem", key);
}

// The sequence protocol (PySequence_Check, reversed, the C bisect) as without the fast path.
PyObject* vector_item(PyObject* self, std::ptrdiff_t i) {
    PyObject* const key = py.PyLong_FromSsize_t(i);
    PyObject* const item = key ? vector_getitem(self, key) : nullptr;
    py.Py_DecRef(key);
    return item;
}

// value is null for del vec[i], which the interface refuses.
int vector_setitem(PyObject* self, PyObject* key, PyObject* value) {
    double* const pd = value && (type_of(value) == py.PyFloat_Type ||
                                 type_of(value) == py.PyLong_Type)
                           ? element(self, key)
                           : nullptr;
    double const d = pd ? py.PyFloat_AsDouble(value) : 0.0;
    if (pd && !(d == -1.0 && py.PyErr_Occurred())) {
        *pd = d;
        return 0;
    }
    py.PyErr_Clear();
    return status(call(self, "_setitem", key, value));
}

/// Where a slot of cls's instances (of size bytes) is: the word that holds what it is set to.
std::ptrdiff_t slot_offset(PyObject* cls, const char* name, std::ptrdiff_t size) {
    PyObject* const probe = py.PyObject_CallMethod(cls, "__new__", "O", cls);
    PyObject* const marker = py.PyFloat_FromDouble(0.5);
    PyObject* const set = probe && marker
                              ? py.PyObject_CallMethod(probe, "__setattr__", "sO", name, marker)
                              : nullptr;
    std::ptrdiff_t found = -1;
    for (std::ptrdiff_t i = 0; set && found < 0 && i < size; i += sizeof marker) {
        found = at(probe, i) == marker ? i : -1;
    }
    for (PyObject* p: {set, probe, marker}) {
        py.Py_DecRef(p);
    }
    return found;
}

/// The base types and, unless NRN_PYTHON_FAST_PATH=0, the fast path's. False on failure.
bool make_types() {
    using stable::t_object_ex, stable::Slot;
    auto const fn = [](auto f) { return reinterpret_cast<void*>(f); };
    bool const fast = fast_path_enabled();
    static stable::Member object_members[]{
        {"_obj", t_object_ex, offsetof(Wrapper, address), 0, nullptr}, {}};
    static stable::Member section_members[]{
        {"_sec", t_object_ex, offsetof(Wrapper, address), 0, nullptr},
        {"_pynative", t_object_ex, offsetof(Wrapper, pynative), 0, nullptr},
        {}};
    static Slot slots[][4]{
        {{stable::tp_dealloc, fn(dealloc<release_object>)}, {stable::tp_members, object_members}},
        {{stable::tp_dealloc, fn(dealloc<release_section>)}, {stable::tp_members, section_members}},
        {{stable::tp_getattro, fn(getattro<true>)}, {stable::tp_setattro, fn(setattro<true>)}},
        {{stable::tp_getattro, fn(getattro<false>)}, {stable::tp_setattro, fn(setattro<false>)}},
        {{stable::mp_subscript, fn(vector_getitem)},
         {stable::mp_ass_subscript, fn(vector_setitem)},
         {stable::sq_item, fn(vector_item)}}};
    constexpr unsigned base = stable::tpflags_basetype;
    static stable::Spec specs[]{{"hoc._Object", sizeof(Wrapper), 0, base, slots[0]},
                                {"nrn._Section", sizeof(Wrapper), 0, base, slots[1]},
                                {"nrn._Segment", 0, 0, base, slots[2]},
                                {"nrn._Mechanism", 0, 0, base, slots[3]},
                                {"hoc._Vector", 0, 0, base, slots[4]}};
    for (int i = 0; i < 5; ++i) {  // None for the fast path's, when it is off
        if (!(types[i] = i < 2 || fast ? py.PyType_FromSpec(&specs[i]) : py._Py_NoneStruct)) {
            return false;
        }
    }
    return true;
}
}  // namespace

void tree_changed() {
    if (config[Epoch] && py.Py_IsInitialized()) {
        int const gil = py.PyGILState_Ensure();
        py.PyList_SetItem(config[Epoch], 0, py.Py_BuildValue(""));  // None
        py.PyGILState_Release(gil);
    }
}

}  // namespace neuron::python

using namespace neuron::python;

// The interface's entries, which take and give Python objects (docs/capi.rst). With the GIL.

/// (object, section, segment, mechanism, vector), the base types, borrowed; None for those of the
/// fast path when it is off. Null with an error set if they could not be made.
extern "C" NRN_EXPORT PyObject* nrn_python_types() {
    if (!start() || (!types[0] && !make_types())) {
        return nullptr;
    }
    return py.Py_BuildValue("(OOOOO)", types[0], types[1], types[2], types[3], types[4]);
}

/// (Segment, Mechanism, Vector, epoch, nseg generations, range variables, mechanisms, 0,
/// AttributeError, Segment.__basicsize__, Mechanism.__basicsize__): the interface's classes and
/// tables. 0, or -1 if they are not all there.
extern "C" NRN_EXPORT int nrn_python_fast_setup(PyObject* objects) {
    static char const* const names[]{
        "_sec", "_x", "_mech_cache", "_seg", "_x", "_name", "_epoch", "_nseg"};
    PyObject* given[11];
    std::ptrdiff_t places[8];
    for (int i = 0; i < 11; ++i) {
        if (!(given[i] = py.PyTuple_GetItem(objects, i))) {
            return -1;
        }
    }
    for (int i = 0; i < 8; ++i) {
        std::ptrdiff_t const size = py.PyLong_AsSsize_t(given[i < 3 ? 9 : 10]);
        if ((places[i] = slot_offset(given[i < 3 ? 0 : 1], names[i], size)) < 0) {
            return -1;
        }
    }
    py.Py_IncRef(objects);  // keeps what it holds
    std::copy(given, given + 9, config);
    std::copy(places, places + 8, offset);
    return 0;
}
