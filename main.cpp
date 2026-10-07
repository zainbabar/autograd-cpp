#include <iostream>
#include <memory>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <functional>
#include <algorithm>
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

// make it easier to create new shit, refactor this to have a wrapper class to make it cleaner
using vp = shared_ptr<Value>;
vp val(double x) { return make_shared<Value>(x); }

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
    // the way the ordering is setup node pushes its gradient back to all its inputs before
    // anyhting else touches it 
    for (auto nodeIt = order.rbegin(); nodeIt != order.rend(); ++nodeIt) {
        shared_ptr<Value>& node = *nodeIt;
        node->backward();
    }
}


// each value object / node is on the heap, it exists once
// so its the same object everywhere its used if we reuse nodes, shared
// so gradients accumulate on one object
// and child nodes keep their inputs alive for backward pass
int main() {
    // testing stuff hidden
    /*
    shared_ptr<Value> x = make_shared<Value>(2);
    shared_ptr<Value> y = make_shared<Value>(3);
    shared_ptr<Value> q = x + y;
    shared_ptr<Value> f = q * x;
    // sketched out:
    // dfdq += 2 -> intermediate graident
    // dqdx += 1 -> local derivative 
    // dfdx += dfdq * dqdx = 2 * 1 = 2 -> current gradient, just passes on, local deriv is 1
    // dqdy += 1 -> intermediate
    // dfdy += dfdq * dqdy = 2 * 1 = 2 -> gradient, again just passes on since add
    // now need to add more to x.grad since its used twice, add the 
    // dfdx += 5 -> final gradient, (since q is 5) 
    // so final gradients are x.grad = 7 (5 + 2), y.grad = 2

    // test this out, manually call .backward on all see if it works
    f->grad = 1;
    f->backward();
    q->backward();
    cout << "x.grad: " << x->grad << ", y.grad: " << y->grad << endl;
    */
    // weights and biases
    vp x1 = val(2);
    vp x2 = val(0);
    vp w1 = val(-3);
    vp w2 = val(1);
    vp b = val(6.8813735870195432);

    vp x1w1 = x1 * w1;
    vp x2w2 = x2 * w2;
    vp dp = x1w1 + x2w2;
    vp n = dp + b;
    vp o = tanh(n);
    o->grad = 1; // set gradient of output 

    // try out using automatic gradient calc now 
    // visited set and ordering for build function
    unordered_set<Value*> visited{};
    vector<shared_ptr<Value>> order{};
    build(o, visited, order);

    backprop(o,order);

    cout << x1->grad << " " << w1->grad << " " << x2->grad << " " << w2->grad << endl;
}
