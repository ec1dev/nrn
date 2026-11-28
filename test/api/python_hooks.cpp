// nrn_python_hook_set's answers, and nrn_hoc_error_is_returned where HOC errors are caught. The
// hooks themselves need Python; test/unit_tests/hoc_python/test_python_hooks.py calls them.
#include "neuronapi.h"
#include <array>
#include <cstring>
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
    if (!failures) {
        std::cout << "OK\n";
    }
    return failures != 0;
}
