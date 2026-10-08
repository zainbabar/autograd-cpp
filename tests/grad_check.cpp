#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "autograd.hpp"
#include "value.hpp"
using namespace std;

// Checks backprop against a numerical derivative.
// f builds the graph from a leaf x and returns the output, called once per evaluation so each gets a fresh graph.
// Returns the relative error between backprop's dx and the numerical dx.
double grad_check(function<shared_ptr<Value>(const shared_ptr<Value>&)> f, double x) {
    // central difference: (f(x+h) - f(x-h)) / 2h, more accurate than one-sided (f(x+h) - f(x)) / h
    double h = 0.0001;
    double dfdx = (f(make_shared<Value>(x + h))->data - f(make_shared<Value>(x-h))->data) / (2 * h);
    shared_ptr<Value> xval = make_shared<Value>(x);
    shared_ptr<Value> output = f(xval);
    backprop(output);

    // Relative error, so the tolerance scales with the size of the gradient:
    // an error of 1e-4 is fine on a grad of 100, but not on a grad of 1e-3.
    // The 1e-8 floor avoids dividing by 0 when both grads are 0.
    double den = max({abs(xval->grad), abs(dfdx), 1e-8});
    return abs(xval->grad - dfdx) / den;
}

int main() {
    cout << fixed << setprecision(6);

    // 1. backprop through a small expression by hand
    cout << "== gradients of c = tanh(a*b + a/b) ==" << endl;
    auto a = make_shared<Value>(0.5);
    auto b = make_shared<Value>(-1.5);
    auto c = tanh(a * b + a / b);
    backprop(c);
    cout << "  c = " << c->data << endl;
    cout << "  dc/da = " << a->grad << endl;
    cout << "  dc/db = " << b->grad << endl;

    // 2. check every op's backward against a numerical derivative, errors should be ~1e-8 or smaller
    cout << "\n== grad check (relative error vs numerical) ==" << endl;
    using Fn = function<shared_ptr<Value>(const shared_ptr<Value>&)>;
    vector<pair<string, Fn>> checks = {
        {"x + 3",            [](const shared_ptr<Value>& x) { return x + 3; }},
        {"2 - x",            [](const shared_ptr<Value>& x) { return 2 - x; }},
        {"x * x",            [](const shared_ptr<Value>& x) { return x * x; }},
        {"1 / x",            [](const shared_ptr<Value>& x) { return 1 / x; }},
        {"x^3",              [](const shared_ptr<Value>& x) { return pow(x, 3); }},
        {"tanh(x)",          [](const shared_ptr<Value>& x) { return tanh(x); }},
        {"tanh(x*x/3) - x",  [](const shared_ptr<Value>& x) { return tanh(x * x / 3) - x; }},
    };
    for (auto& [name, f] : checks) {
        cout << "  " << left << setw(18) << name << right << scientific << setprecision(2)
             << grad_check(f, 0.7) << fixed << setprecision(6) << endl;
    }
}
