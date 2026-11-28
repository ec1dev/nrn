// nrn_segment_point_processes: the point processes of a node, newest first, as Python's
// seg.point_processes() lists them.
#include <array>
#include <cmath>
#include <iostream>
#include "neuronapi.h"

extern "C" void modl_reg(){/* No modl_reg */};

static int failures;

static void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// A new point process of class name at x of the section (which is the accessed one).
static Object* place(const char* name, double x) {
    nrn_double_push(x);
    return nrn_object_new(nrn_symbol(name), 1);
}

int main() {
    std::array<const char*, 4> args = {"segment_point_processes", "-nogui", "-nopython", nullptr};
    nrn_init(3, args.data());
    Section* soma = nrn_section_new("soma");
    Section* dend = nrn_section_new("dend");
    nrn_nseg_set(soma, 3);
    nrn_section_connect(dend, 0, soma, 1);
    nrn_section_push(soma);
    Object* first = place("IClamp", 0.5);
    Object* second = place("ExpSyn", 0.5);
    Object* end = place("IClamp", 1);
    Object* other = place("IClamp", 0.1);
    nrn_section_pop();
    nrn_section_push(dend);
    Object* child = place("ExpSyn", 0);
    nrn_section_pop();
    nrn_tree_setup(soma, 1);

    std::array<Object*, 4> found{};
    check(nrn_segment_point_processes(soma, 0.5, found.data(), 4) == 2, "two at 0.5");
    check(found[0] == second && found[1] == first, "newest first");
    check(nrn_segment_point_processes(soma, 0.5, nullptr, 0) == 2, "count only");
    found = {};
    check(nrn_segment_point_processes(soma, 0.5, found.data(), 1) == 2 && found[0] == second &&
              !found[1],
          "fills max");
    check(nrn_segment_point_processes(soma, 0.1, found.data(), 4) == 1 && found[0] == other,
          "another node");
    check(nrn_segment_point_processes(soma, 0.9, nullptr, 0) == 0, "none");
    found = {};
    // The node the child connects to is the parent's 1 end.
    check(nrn_segment_point_processes(soma, 1, found.data(), 4) == 2 && found[0] == child &&
              found[1] == end,
          "a shared node lists both sections'");
    check(nrn_segment_point_processes(dend, 0, nullptr, 0) == 2, "from the child too");
    check(nrn_segment_point_processes(soma, NAN, nullptr, 0) == -1, "NaN");
    check(nrn_segment_point_processes(soma, 1.5, nullptr, 0) == -1, "x > 1");
    check(nrn_segment_point_processes(nullptr, 0.5, nullptr, 0) == -1, "no section");
    for (Object* o: {first, second, end, other, child}) {
        nrn_object_unref(o);
    }
    check(nrn_segment_point_processes(soma, 0.5, nullptr, 0) == 0, "freed");
    if (!failures) {
        std::cout << "OK\n";
    }
    return failures != 0;
}
