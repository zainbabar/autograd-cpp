#include <iostream>
#include <memory>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <functional>
#include <random>
#include <string>
using namespace std;


// TODO: 
// add subtraction, negation exp, pow 

// our ops for tracking backward pass
enum class Op {NONE, ADD, MULT, TANH};

class Value {
  public:
    double data; // data
    vector<shared_ptr<Value>> inputs;  // the inputs that led to this current node
    double grad; // gradient
    Op op;
    function<void()> backward; // each nodes backward pushes its own grad to its own inputs

    Value(   
        double data, 
        vector<shared_ptr<Value>> inputs=vector<shared_ptr<Value>>{}, // empty by def 
        Op op=Op::NONE  // none by def
    ): data{data}, inputs{inputs}, grad{0}, op{op}, backward([]{}) {} 

};
shared_ptr<Value> operator+(const shared_ptr<Value>& self, const shared_ptr<Value>& other) {
    shared_ptr<Value> out = make_shared<Value>(
        self->data + other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::ADD
    );
    // lambda, so when operation is done the corresponding backward def is set to calc gradients correctly
    // hold ref to its inputs 
    // local deriv * incoming gradient, update its inputs gradients
    out->backward = [o = out.get()] {
        for (auto input : o->inputs) {
            input->grad += 1 * o->grad;
        }
    };
    return out;
}

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
    // let the lambda caputre a raw ptr to the output 
    // the output has all the info that we need
    // so the lambda can actually access the stuff to work on 
    // let lambda know where to look, o is lambdas way back to node it lives in 
    // cuz self, other, out are all in the stack frame and get destroyed
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
    // think of this is self/other, dself = 1/other, dother = self/other^2
    shared_ptr<Value> out = make_shared<Value>(
        self->data / other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::MULT 
    );
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += (1 / o->inputs[1]->data) * o->grad;
        o->inputs[1]->grad += (o->inputs[0]->data / pow(o->inputs[0]->data, 2)) * o->grad;
    };
    return out;
}

shared_ptr<Value> operator/(const shared_ptr<Value>& self, double other) {
    return self / make_shared<Value>(other);
}

shared_ptr<Value> operator/(double other, const shared_ptr<Value>& self) {
    return self / other;
}

// tanh funciton as our activiation function for now 
shared_ptr<Value> tanh(const shared_ptr<Value>& self) {
    double x = self->data;
    // huge values of x can cause overflow here, so need to keep exponent negative
    double t = 0;
    if (x < 0) { t = (exp(2*x) - 1) / (exp(2*x) + 1); } // actual tanh value for out 
    else { t = (1 - exp(2*(-x))) / (1 + exp(2*(-x))); } // actual tanh value for out
    shared_ptr<Value> out = make_shared<Value>(t, vector<shared_ptr<Value>>{self}, Op::TANH);
    // set our outs gradient, so take local deriv of its single input, and mult by incoming gradient
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += (1 - pow(o->data, 2)) * o->grad;
    };
    return out;
}

// nodes backward pushes its gradient down to only its inputs
// now we want to make a funciton that can call it in order instead of manually, only 1 .backward needed
// build ordering using topological sort funciton

// takes output node and returns list in forward order, every node comes striclty after its inputs
// call on node to visit it, if alr in visited return (avoid duplicates)
// if not visted add it to visited and visit (recursive call) every node in its inputs 
// then add node to list
// this makes it so node is only added to list after all its inputs have been added to list

// recursive helper: visited and order are shared across every call
void visit(const shared_ptr<Value>& node,
           unordered_set<Value*>& visited,
           vector<shared_ptr<Value>>& order) {
    if (visited.contains(node.get())) { return; } // already visited
    visited.insert(node.get());
    for (const auto& input : node->inputs) {      // no-op if it has none
        visit(input, visited, order);
    }
    // only after all inputs have been added do we add this node
    order.push_back(node);
}

// entry point: creates the shared state once and returns the order
vector<shared_ptr<Value>> build(const shared_ptr<Value>& output) {
    unordered_set<Value*> visited;
    vector<shared_ptr<Value>> order;
    visit(output, visited, order);
    return order;
}

// note we wanna do our backward calc in REVERSE order of this list, start from output and go backwards
void backprop(const shared_ptr<Value>& output) {
    vector<shared_ptr<Value>> order = build(output);
    output->grad = 1;
    // iterate from end, deref it to get node and call its backward
    // way the ordering is setup means that a node has its complete gradient before pushing back to its inputs
    // since all nodes that used it as an input have alr pushed back gradient
    for (auto nodeIt = order.rbegin(); nodeIt != order.rend(); ++nodeIt) {
        shared_ptr<Value>& node = *nodeIt;
        node->backward();
    }
}

// takes in a function (our calcution) f that takes a variable leaf node, does forward pass
// and returns the output calculated
// x is the variable in question
// compute the numerical gradient of x and compare against a backprop pass grad, return the diff
double grad_check(function<shared_ptr<Value>(const shared_ptr<Value>&)> f, double x) {
    // compute numerical gradient, traditional derivative calc 
    double h = 0.0001;
    double dfdx = (f(make_shared<Value>(x + h))->data - f(make_shared<Value>(x-h))->data) / (2 * h);
    // now do our normal forward pass / backprop
    shared_ptr<Value> xval = make_shared<Value>(x);
    shared_ptr<Value> output = f(xval);
    backprop(output); 

    // use relative error, so we can see how big the error is in terms of the size of the gradient
    // big gradient, small error ok, small gradient small error not ok
    double den = max({abs(xval->grad), abs(dfdx), 1e-8});
    return abs(xval->grad - dfdx) / den;
}

// each value object / node is on the heap, it exists once
// so its the same object everywhere its used if we reuse nodes, shared
// so gradients accumulate on one object
// and child nodes keep their inputs alive for backward pass


class Neuron {
  public:
    // weights and biases are Value obj, since we want to see their gradient and 
    // adjust accordinly in training 
    unsigned long n_inputs;
    vector<shared_ptr<Value>> weights;
    shared_ptr<Value> bias;

    Neuron(unsigned long n): n_inputs{n} {
        // setup random dist of nums, use reproducible seed if we want
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-1, 1);
        // populate weights and biases for this neuron
        for (int i = 0; i < n_inputs; ++i) {
            weights.push_back(make_shared<Value>(dist(gen)));
        }
        bias = make_shared<Value>(dist(gen));
    }
    // takes vector of input nodes, and returns dot product + tanh activation function as a new Value
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