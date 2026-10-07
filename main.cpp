#include <iostream>
#include <memory>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <functional>
#include <random>
#include <string>
using namespace std;


// TODO: add subtraction, negation, exp, pow

// which op produced a node (NONE for leaves)
enum class Op {NONE, ADD, MULT, DIV, TANH};

// Every node lives on the heap behind a shared_ptr, so reusing a node in several ops means
// they all point at the same object and its grad accumulates in one place.
// Each node also holds shared_ptrs to its inputs, which keeps the whole graph alive
// for the backward pass as long as the output is alive.
class Value {
  public:
    double data;
    vector<shared_ptr<Value>> inputs;  // nodes this one was computed from (empty for leaves)
    double grad;
    Op op;
    // pushes this node's grad down into its inputs' grads, set by whichever op created the node
    function<void()> backward;

    Value(
        double data,
        vector<shared_ptr<Value>> inputs=vector<shared_ptr<Value>>{},
        Op op=Op::NONE
    ): data{data}, inputs{inputs}, grad{0}, op{op}, backward([]{}) {}

};
shared_ptr<Value> operator+(const shared_ptr<Value>& self, const shared_ptr<Value>& other) {
    shared_ptr<Value> out = make_shared<Value>(
        self->data + other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::ADD
    );
    // The lambda captures a raw pointer to out, not a reference or a shared_ptr:
    // - self/other/out are locals, so capturing by reference would dangle once this returns
    // - a shared_ptr copy would make out own itself (ref cycle), so it would never be freed
    // The lambda is stored inside *o, so o is always valid whenever it runs.
    // d(a+b)/da = d(a+b)/db = 1, so each input just gets the incoming grad.
    // += since a node can feed several outputs, its grad is the sum over all of them (also handles a + a)
    out->backward = [o = out.get()] {
        for (auto input : o->inputs) {
            input->grad += 1 * o->grad;
        }
    };
    return out;
}

// scalar overloads: wrap the double in a leaf Value and reuse the Value version
shared_ptr<Value> operator+(const shared_ptr<Value>& self, double other) {
    return self + make_shared<Value>(other);
}

shared_ptr<Value> operator+(double other, const shared_ptr<Value>& self) {
    return self + other;
}

shared_ptr<Value> operator*(const shared_ptr<Value>& self, const shared_ptr<Value>& other) {
    shared_ptr<Value> out = make_shared<Value>(
        self->data * other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::MULT
    );
    // d(a*b)/da = b and d(a*b)/db = a, so each input's grad is the other input's value * incoming grad
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += o->inputs[1]->data * o->grad;
        o->inputs[1]->grad += o->inputs[0]->data * o->grad;
    };
    return out;
}

shared_ptr<Value> operator*(const shared_ptr<Value>& self, double other) {
    return self * make_shared<Value>(other);
}

shared_ptr<Value> operator*(double other, const shared_ptr<Value>& self) {
    return self * other;
}

shared_ptr<Value> operator/(const shared_ptr<Value>& self, const shared_ptr<Value>& other) {
    shared_ptr<Value> out = make_shared<Value>(
        self->data / other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::DIV
    );
    // d(a/b)/da = 1/b and d(a/b)/db = -a/b^2
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += (1 / o->inputs[1]->data) * o->grad;
        o->inputs[1]->grad += -(o->inputs[0]->data / pow(o->inputs[1]->data, 2)) * o->grad;
    };
    return out;
}

shared_ptr<Value> operator/(const shared_ptr<Value>& self, double other) {
    return self / make_shared<Value>(other);
}

shared_ptr<Value> operator/(double other, const shared_ptr<Value>& self) {
    return make_shared<Value>(other) / self;
}

// tanh as our activation function for now
shared_ptr<Value> tanh(const shared_ptr<Value>& self) {
    double x = self->data;
    // tanh(x) = (e^2x - 1) / (e^2x + 1), but for large positive x e^2x overflows to inf and gives inf/inf = NaN.
    // For x >= 0, use the same formula multiplied through by e^-2x so the exponent is never positive.
    double t = 0;
    if (x < 0) { t = (exp(2*x) - 1) / (exp(2*x) + 1); }
    else { t = (1 - exp(2*(-x))) / (1 + exp(2*(-x))); }
    shared_ptr<Value> out = make_shared<Value>(t, vector<shared_ptr<Value>>{self}, Op::TANH);
    // d tanh(x)/dx = 1 - tanh(x)^2, and tanh(x) is already stored in o->data
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += (1 - pow(o->data, 2)) * o->grad;
    };
    return out;
}

// Each node's backward only pushes grad one level down to its inputs, so to backprop the whole
// graph with a single call we need to run them in the right order.
// Topological sort: DFS that adds a node to the list only after all of its inputs are added,
// so every node comes strictly after its inputs.

// recursive helper: visited and order are shared across every call
void visit(const shared_ptr<Value>& node,
           unordered_set<Value*>& visited,
           vector<shared_ptr<Value>>& order) {
    // a node can be reached through more than one path, only add it once
    if (visited.contains(node.get())) { return; }
    visited.insert(node.get());
    for (const auto& input : node->inputs) {
        visit(input, visited, order);
    }
    order.push_back(node);
}

// entry point: creates the shared state once and returns the order
vector<shared_ptr<Value>> build(const shared_ptr<Value>& output) {
    unordered_set<Value*> visited;
    vector<shared_ptr<Value>> order;
    visit(output, visited, order);
    return order;
}

// Grads aren't reset first, so calling this twice on the same graph accumulates.
void backprop(const shared_ptr<Value>& output) {
    vector<shared_ptr<Value>> order = build(output);
    output->grad = 1; // d(output)/d(output), seeds the chain rule
    // Walk the order in reverse (output first). By the time a node's backward runs, every node
    // that used it as an input has already pushed its grad in, so its grad is complete.
    for (auto nodeIt = order.rbegin(); nodeIt != order.rend(); ++nodeIt) {
        shared_ptr<Value>& node = *nodeIt;
        node->backward();
    }
}

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


class Neuron {
  public:
    unsigned long n_inputs;
    // weights and bias are Values so backprop gives them grads we can use to update them in training
    vector<shared_ptr<Value>> weights;
    shared_ptr<Value> bias;

    Neuron(unsigned long n): n_inputs{n} {
        // seeded from random_device, swap rd() for a fixed number to get reproducible runs
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-1, 1);
        for (int i = 0; i < n_inputs; ++i) {
            weights.push_back(make_shared<Value>(dist(gen)));
        }
        bias = make_shared<Value>(dist(gen));
    }
    // returns tanh(w·x + b) as a new Value
    // if inputs and weights differ in length, only the first min of the two are used
    shared_ptr<Value> operator()(const vector<shared_ptr<Value>>& inputs) {
        shared_ptr<Value> dot = make_shared<Value>(0);
        for (int i = 0; i < min(inputs.size(), n_inputs); ++i) {
            dot = dot + (weights[i] * inputs[i]);
        }
        dot = dot + bias;
        shared_ptr<Value> output = tanh(dot);
        return output;
    }
};

bool approx(double a, double b) { return fabs(a - b) < 1e-4; }
void check(const string& name, bool ok) { cout << (ok ? "PASS  " : "FAIL  ") << name << endl; }

void test_neuron_fixed() {
    vector<shared_ptr<Value>> xs = { make_shared<Value>(2.0), make_shared<Value>(3.0), make_shared<Value>(-1.0) };
    Neuron n{3};
    n.weights[0]->data = 0.5;
    n.weights[1]->data = -0.2;
    n.weights[2]->data = 0.3;
    n.bias->data = 0.1;

    shared_ptr<Value> out = n(xs);
    backprop(out);

    // w·x + b = 1 - 0.6 - 0.3 + 0.1 = 0.2, and the tanh local grad is 1 - tanh(0.2)^2 ≈ 0.9610,
    // so dw_i = x_i * 0.9610, dx_i = w_i * 0.9610, db = 0.9610
    check("fixed: out = tanh(0.2)", approx(out->data, 0.19738));
    check("fixed: dw0 = 1.9221",   approx(n.weights[0]->grad, 1.9221));
    check("fixed: dw1 = 2.8831",   approx(n.weights[1]->grad, 2.8831));
    check("fixed: dw2 = -0.9610",  approx(n.weights[2]->grad, -0.9610));
    check("fixed: db = 0.9610",    approx(n.bias->grad, 0.9610));
    check("fixed: dx0 = 0.4805",   approx(xs[0]->grad, 0.4805));
    check("fixed: dx1 = -0.1922",  approx(xs[1]->grad, -0.1922));
    check("fixed: dx2 = 0.2883",   approx(xs[2]->grad, 0.2883));
}

void test_neuron_random() {
    vector<shared_ptr<Value>> xs = { make_shared<Value>(2.0), make_shared<Value>(3.0), make_shared<Value>(-1.0) };
    Neuron n{3};
    shared_ptr<Value> out = n(xs);
    backprop(out);

    check("random: out in (-1, 1)", out->data > -1 && out->data < 1);
    bool all_nonzero = (n.bias->grad != 0);
    for (const auto& w : n.weights) if (w->grad == 0) all_nonzero = false;
    check("random: every weight and bias has nonzero grad", all_nonzero);
}

int main() {
    test_neuron_fixed();
    test_neuron_random();
}
