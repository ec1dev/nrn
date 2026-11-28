// nrn_python_hook_set's answers, nrn_hoc_error_is_returned where HOC errors are caught, the
// section owner hooks left out for HOC's sections, what a PythonObject holds, and
// nrn_hoc_readline. The hooks themselves need Python;
// test/unit_tests/hoc_python/test_python_hooks.py calls them.
#include "neuronapi.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>

extern "C" void modl_reg() {}

static int failures;

static void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

static int returned = -1;

static void probe() {
    returned = nrn_hoc_error_is_returned();
    nrn_hoc_ret();
    nrn_double_push(0.);
}

static void hook() {}

static int owner_calls;

static int section_cell(void*, Object** result) {
    ++owner_calls;
    *result = nullptr;
    return 1;
}

static int section_cell_equals(void*, Object*) {
    ++owner_calls;
    return 0;
}

int main() {
    std::array<const char*, 4> args = {"python_hooks", "-nogui", "-nopython", nullptr};
    nrn_init(3, args.data());

    char error[64] = "unchanged";
    check(nrn_python_hook_set("praxis", hook, error, sizeof error) == 0, "set a hook");
    check(std::strcmp(error, "unchanged") == 0, "no message on success");
    check(nrn_python_hook_set("praxis", nullptr, error, sizeof error) == 0, "clear a hook");
    check(nrn_python_hook_set("nosuch", hook, error, sizeof error) == 1, "an unknown name");
    check(std::strcmp(error, "nosuch is not a Python hook") == 0,
          std::string{"its message: "} + error);
    check(nrn_python_hook_set(nullptr, hook, error, sizeof error) == 1, "no name");
    check(std::strcmp(error, "a null name is not a Python hook") == 0,
          std::string{"its message: "} + error);
    check(nrn_python_hook_set("nosuch", hook, nullptr, 0) == 1, "no buffer");

    nrn_register_function(probe, "probe", 280);
    check(nrn_hoc_error_is_returned() == 0, "outside HOC");
    check(nrn_hoc_call_nothrow("probe()") == 0 && returned == 1, "in nrn_hoc_call_nothrow");
    returned = -1;
    check(nrn_function_call_nothrow(nrn_symbol("probe"), 0, error, sizeof error) == 0 &&
              returned == 1,
          "in nrn_function_call_nothrow");
    nrn_double_pop();
    returned = -1;
    check(nrn_hoc_call_nothrow("{execute1(\"probe()\")}") == 0 && returned == 0,
          "in execute1 inside nrn_hoc_call_nothrow");
    returned = -1;
    check(nrn_hoc_call("probe()") == 0 && returned == 0, "in nrn_hoc_call");
    check(nrn_hoc_error_is_returned() == 0, "after the calls");

    // The section owner hooks are not asked about a section HOC made (it has no owner).
    check(nrn_python_hook_set("section_cell",
                              reinterpret_cast<void (*)()>(section_cell),
                              error,
                              sizeof error) == 0 &&
              nrn_python_hook_set("section_cell_equals",
                                  reinterpret_cast<void (*)()>(section_cell_equals),
                                  error,
                                  sizeof error) == 0,
          "set the section hooks");
    for (char const* const statement: {"create plain",
                                       "objref nil, syn, nc, cells",
                                       "plain syn = new ExpSyn(.5)",
                                       "nc = new NetCon(nil, syn)",
                                       "cells = nc.postcelllist()",
                                       "plain cells = section_owner()"}) {
        check(nrn_hoc_call_nothrow(statement) == 0, statement);
    }
    check(owner_calls == 0, "no owner hook for a HOC section");
    nrn_python_hook_set("section_cell", nullptr, error, sizeof error);
    nrn_python_hook_set("section_cell_equals", nullptr, error, sizeof error);

    // A PythonObject holds what it was given; without a release hook it only lets go of it.
    int held{};
    void* value = &failures;
    Object* const o = nrn_python_object_new(&held);
    check(nrn_python_object_get(o, &value) == 1 && value == &held, "what a PythonObject holds");
    Object* const vector = nrn_object_new(nrn_symbol("Vector"), 0);
    value = &failures;
    check(nrn_python_object_get(vector, &value) == 0 && value == &failures, "not a PythonObject");
    check(nrn_python_object_get(nullptr, &value) == 0, "no object");
    nrn_object_unref(vector);
    nrn_object_unref(o);

    // HOC's reader, of stdin here: a line, the last one, then "" at the end.
    if (std::FILE* const input = std::fopen("python_hooks_input.txt", "w")) {
        std::fputs("first line\nlast", input);
        std::fclose(input);
    }
    check(std::freopen("python_hooks_input.txt", "r", stdin) != nullptr, "stdin from a file");
    char const* line = nrn_hoc_readline("prompt> ");
    check(line && std::strcmp(line, "first line\n") == 0, "a line");
    line = nrn_hoc_readline("prompt> ");
    check(line && std::strcmp(line, "last\n") == 0,
          std::string{"the last line: "} + (line ? line : "NULL"));
    line = nrn_hoc_readline("prompt> ");
    check(line && !*line, "the end of the input");
    if (!failures) {
        std::cout << "OK\n";
    }
    return failures != 0;
}
