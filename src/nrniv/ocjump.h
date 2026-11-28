#pragma once
#include <cstddef>
#include <cstring>
#include <exception>
#ifdef __GLIBCXX__
#include <cxxabi.h>
#endif

union Inst;
struct Object;
union Objectdata;
struct Symlist;

/** @brief How many NEURON try { ... } catch(...) { ... } blocks are in the call
 * stack.
 *
 *  Errors inside NEURON are triggered using hoc_execerror, which ultimately
 *  throws an exception. To replicate the old logic, we sometimes need to insert
 *  a try/catch block *only if* there is no try/catch block less deeply nested
 *  on the call stack. This global variable tracks how many such blocks are
 *  currently present on the stack.
 */
extern int nrn_try_catch_nest_depth;

/** @brief Helper type for incrementing/decrementing nrn_try_catch_nest_depth.
 */
struct try_catch_depth_increment {
    try_catch_depth_increment() {
        ++nrn_try_catch_nest_depth;
    }
    ~try_catch_depth_increment() {
        --nrn_try_catch_nest_depth;
    }
};

namespace neuron::python {
/// A place where HOC errors are caught: whether its caller gets the error as a status and message
/// (a _nothrow function of the C API) or HOC itself goes on (nrn_hoc_error_is_returned).
class ErrorScope {
    ErrorScope* const outer_;

  public:
    bool const can_be_taken;
    explicit ErrorScope(bool can_be_taken);
    ~ErrorScope();
    ErrorScope(const ErrorScope&) = delete;
    ErrorScope& operator=(const ErrorScope&) = delete;
    static ErrorScope* innermost();
};
}  // namespace neuron::python

/// Call f. False if it threw anything; error is the text then. The unwinding of a cancelled thread
/// is not an error and goes on.
template <typename F>
bool nrn_nothrow(F const& f, char* error = nullptr, std::size_t size = 0) {
    auto const report = [&](char const* text) {
        if (error && size) {
            std::strncpy(error, text, size - 1);
            error[size - 1] = '\0';
        }
        return false;
    };
    if (error && size) {
        error[0] = '\0';
    }
    try {
        f();
        return true;
#ifdef __GLIBCXX__
    } catch (abi::__forced_unwind const&) {
        throw;
#endif
    } catch (std::exception const& e) {
        return report(e.what());
    } catch (...) {
        return report("Unknown exception occurred");
    }
}

struct ObjectContext {
    ObjectContext(Object*);
    ~ObjectContext();

  private:
    Object* a1;
    Objectdata* a2;
    int a4;
    Symlist* a5;
};

struct OcJump {
    static bool execute(Inst* p);
    static bool execute(const char*, Object* ob = NULL);
    static void* fpycall(void* (*) (void*, void*), void*, void*);
    static void execute_throw_on_exception(Symbol* sym, int narg);
    static void execute_throw_on_exception(Object* obj, Symbol* sym, int narg);
    static Object* newobj_throw_on_exception(Symbol* sym, int narg);
};
