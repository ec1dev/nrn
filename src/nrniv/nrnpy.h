#pragma once
#include "nrndlldef.h"
#include <cstdint>
#include <string>
#include <vector>
/**
 * The functions with which HOC and the simulator call Python. nrnpy_hooks.cpp
 * sets most of them when NEURON starts; they call what the interface registers
 * (nrn_python_hook_set).
 */
struct Object;
// methods.hoccommand_exec, for MOD VERBATIM (e.g. test/rxd/beforestep_py.mod), which imports this
// from nrniv.dll.
extern NRN_DLLSYM int (*nrnpy_hoccommand_exec)(Object*);
struct Section;
struct Symbol;
class IvocVect;
namespace neuron::python {
struct impl_ptrs {
    Object* (*callable_with_args)(Object*, int narg){};
    double (*call_func)(Object*, int, int*){};
    std::vector<char> (*call_picklef)(const std::vector<char>&, int narg){};
    void (*call_python_with_section)(Object*, Section*){};
    void (*cmdtool)(Object*, int type, double x, double y, int kd){};
    int (*guigetstr)(Object*, char**){};
    double (*guigetval)(Object*){};
    Object** (*gui_helper)(const char* name, Object* obj){};
    Object** (*gui_helper3)(const char* name, Object* obj, int handle_strptr){};
    char** (*gui_helper3_str)(const char*, Object*, int){};
    void (*guisetval)(Object*, double){};
    int (*hoccommand_exec)(Object*){};
    int (*hoccommand_exec_strret)(Object*, char*, int){};
    void (*hoc_nrnpython)(){};
    void (*interpreter_set_path)(){};
    void (*interpreter_finalize)(){};
    int (*interpreter_run)(){};
    Object* (*mpi_alltoall_type)(int, int){};
    double (*object_to_double)(Object*){};
    void* (*opaque_obj2pyobj)(Object*){};
    Object* (*pickle2po)(const std::vector<char>&){};
    std::vector<char> (*po2pickle)(Object*){};
    double (*praxis_efun)(Object* pycallable, Object* hvec){};
    int (*pysame)(Object* o1, Object* o2){};
    /// Windows: wait(arg), while Python's other threads go on (the GUI thread calls Python).
    void (*without_gil)(void (*wait)(void*), void* arg){};
    // Such a common pattern it gets a wrapper
    Object** try_gui_helper(const char* name, Object* obj) const {
        if (gui_helper) {
            return gui_helper(name, obj);
        } else {
            return nullptr;
        }
    }
};
extern impl_ptrs methods;

// What nrnpy_capi.cpp and nrnpy_launch.cpp do with CPython, for the rest of NEURON.
/// Is Python running? Starts it if NEURON may (nrniv), the first time; false once it has been
/// finalized.
bool start();
/// HOC's quit(): finalize Python, in the thread that brought NEURON and Python together.
void finalize();
/// Start Python, when NEURON, not Python, is the main program (nrnpy.cpp).
extern bool (*launch)();
/// A function or variable of the Python in library (a dlopen handle), or of the one this process
/// has loaded if it is null; null if there is none (nrnpy.cpp).
void* python_symbol(void* library, const char* name);
/// Import the neuron package, which registers the interface's hooks. Set by the launcher.
extern bool (*import_interface)();
/// NRN_PYTHON_FAST_PATH is not "0" (exactly, as NEURON compares its other switches).
bool fast_path_enabled();
/// The model may have changed: what the fast path keeps about it is out of date (nrnpy_fast.cpp).
void tree_changed();

// And what nrnpy_hooks.cpp does for them.
/// Python is about to be finalized: no hook is called any more.
void clear_hooks();
/// What neuron/_launch.py does for nrniv -python: run_file, run_command (-c), prompt (NEURON's
/// reader, also for input()), interact, set_path, import_neuron. False if it failed (it printed
/// the error) or is not there.
bool launcher_hook(const char* what, const char* arg);
}  // namespace neuron::python

extern NRN_DLLSYM Symbol* nrnpy_pyobj_sym_;
extern NRN_DLLSYM std::string nrnpy_pyexe;
extern NRN_DLLSYM int nrnpy_nositeflag;
extern NRN_DLLSYM int hoc_max_builtin_class_id;
extern NRN_DLLSYM int (*p_nrnpy_pyrun)(const char*);
extern NRN_DLLSYM int (*nrnpy_pr_stdoe_callback)(int, char*);
extern NRN_DLLSYM void (*nrnpy_sectionlist_helper_)(void*, Object*);
extern NRN_DLLSYM void* (*nrnpy_get_pyobj)(Object*);
extern NRN_DLLSYM void (*nrnpy_restore_savestate)(int64_t, char*);
extern NRN_DLLSYM void (*nrnpy_store_savestate)(char**, uint64_t*);
extern NRN_DLLSYM void (*nrnpy_decref)(void*);
extern NRN_DLLSYM void (*nrnpy_vec_apply_p_)(IvocVect*, Object* f, int start, int end);
extern NRN_DLLSYM int (*nrnpy_call_obj_method)(Object*, const char*, Object*);
extern NRN_DLLSYM int (*nrnpy_call_obj_method_double)(Object*, const char*, double);
extern NRN_DLLSYM IvocVect* (*nrnpy_vec_from_python_p_)(void*);
extern NRN_DLLSYM Object** (*nrnpy_vec_to_python_p_)(void*);
extern NRN_DLLSYM Object** (*nrnpy_vec_as_numpy_helper_)(int, double*);
extern NRN_DLLSYM Object* (*nrnpy_rvp_rxd_to_callable)(Object*);
extern NRN_DLLSYM int (*nrnpy_nrncore_enable_value_p_)();
extern NRN_DLLSYM int (*nrnpy_nrncore_file_mode_value_p_)();
extern NRN_DLLSYM char* (*nrnpy_nrncore_arg_p_)(double);
extern NRN_DLLSYM int (*nrnpy_ob_is_seg)(Object*);
extern NRN_DLLSYM Object* (*nrnpy_seg_from_sec_x)(Section*, double);
extern NRN_DLLSYM Section* (*nrnpy_o2sec_p_)(Object*);
extern NRN_DLLSYM void (*nrnpy_o2loc_p_)(Object*, Section**, double*);
extern NRN_DLLSYM void (*nrnpy_o2loc2_p_)(Object*, Section**, double*);
extern NRN_DLLSYM char* (*nrnpy_pysec_name_p_)(Section*);
extern NRN_DLLSYM Object* (*nrnpy_pysec_cell_p_)(Section*);
extern NRN_DLLSYM int (*nrnpy_pysec_cell_equals_p_)(Section*, Object*);
