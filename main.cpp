#include <iostream>
#include <iterator>
#include <memory>
#include <cmath>
#include <vector>
using namespace std;


// DONE:
// - forward pass enigne, add mult ops, fixed ownereship
// - backward step for add and mult as a switch on op
// - each node step does only its local contribution to its INPUTS w/ += 
// NEXT:
// - write an ordering funciton
// - ordering function returns vector w/ each nodes inputs before the node

// our ops for tracking backward pass
enum class Op {NONE, ADD, MULT, TANH};

class Value {
  public:
    double data; // data
    vector<shared_ptr<Value>> inputs;  // the inputs that led to this current node
    double grad; // gradient
    Op op;
    function<void()> backward;

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

// each value object / node is on the heap, it exists once
// so its the same object everywhere its used if we reuse nodes, shared
// so gradients accumulate on one object
// and child nodes keep their inputs alive for backward pass
int main() {
    shared_ptr<Value> x = make_shared<Value>(2);
    shared_ptr<Value> y = make_shared<Value>(3);
    shared_ptr<Value> q = x + y;
    shared_ptr<Value> r = x * q;
    shared_ptr<Value> f = tanh(r);
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
    r->backward();
    q->backward();
    cout << "x.grad: " << x->grad << ", y.grad: " << y->grad << endl;
    shared_ptr<Value> test = make_shared<Value>(400);
    shared_ptr<Value> out = tanh(test);
    cout << out->data << endl;
}
