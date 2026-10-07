#include <iostream>
#include <memory>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <functional>
using namespace std;

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
void build(const shared_ptr<Value>& node, unordered_set<Value*>& visited, vector<shared_ptr<Value>>& order) {
    if (visited.contains(node.get()) == true) { return; } // node alr visited
    else { // not visited
        visited.insert(node.get()); // add to visited set
        for (const auto &input : node->inputs) { // visit all its inputs, no-op if has none
            build(input, visited, order);
        }
        // only after all inputs have been visited and added to list we add this node
        order.push_back(node);
    }
}
// note we wanna do our backward calc in REVERSE order of this list, start from output and go backwards
void backprop(const shared_ptr<Value>& output, vector<shared_ptr<Value>>& order) {
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
    vector<shared_ptr<Value>> order{};
    unordered_set<Value*> visited{};
    build(output, visited, order);
    backprop(output, order);

    // use relative error, so we can see how big the error is in terms of the size of the gradient
    // big gradient, small error ok, small gradient small error not ok
    double den = max({abs(xval->grad), abs(dfdx), 1e-8});
    return abs(xval->grad - dfdx) / den;
}

// each value object / node is on the heap, it exists once
// so its the same object everywhere its used if we reuse nodes, shared
// so gradients accumulate on one object
// and child nodes keep their inputs alive for backward pass

int main() {
   // function for testing, builds out the graph 
    auto f = [](const shared_ptr<Value>& x) {
        shared_ptr<Value> a = 2 * x;
        shared_ptr<Value> b = a + 1;
        shared_ptr<Value> c = 3 * a;
        return b + c;
    };

    double err = grad_check(f, 2);
    cout << err << endl;

    // test grad_check on a non linear functions 
    auto cubic  = [](const shared_ptr<Value>& x) { return x * x * x; };
    auto neuron = [](const shared_ptr<Value>& x) { return tanh(2 * x + (-1)); };

    cout << "cubic  rel err: " << grad_check(cubic, 1.5) << endl;
    cout << "neuron rel err: " << grad_check(neuron, 1)  << endl;
}
