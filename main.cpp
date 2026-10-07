#include <iostream>
#include <iterator>
#include <memory>
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
enum class Op {NONE, ADD, MULT};

class Value {
  public:
    double data; // data
    vector<shared_ptr<Value>> inputs;  // the inputs that led to this current node
    double grad; // gradient
    Op op;
    function<void()> backward = []{};

    Value(   
        double data, 
        vector<shared_ptr<Value>> inputs=vector<shared_ptr<Value>>{}, // empty by def 
        Op op=Op::NONE  // none by def
    ): data{data}, inputs{inputs}, grad{0}, op{op} {} 

};
shared_ptr<Value> operator+(const shared_ptr<Value>& self, const shared_ptr<Value>& other) {
    shared_ptr<Value> out = make_shared<Value>(
        self->data + other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::ADD
    );
    out->backward = [self, other, out] {
        self->grad += 1 * out->grad;
         other->grad += 1 * out->grad;
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
    out->backward = [self, other, out] {
        self->grad += other->data * out->grad; 
        other->grad += self->data * out->grad; 
    };
    return out;
}

shared_ptr<Value> operator*(const shared_ptr<Value>& self, double other) {
    return self * make_shared<Value>(other);
}

shared_ptr<Value> operator*(double other, const shared_ptr<Value>& self) {
    return self * other;
}

// each value object / node is on the heap, it exists once
// so its the same object everywhere its used if we reuse nodes, shared
// so gradients accumulate on one object
// and child nodes keep their inputs alive for backward pass
int main() {
    shared_ptr<Value> x = make_shared<Value>(2);
    shared_ptr<Value> y = make_shared<Value>(3);
    shared_ptr<Value> q = x + y;
    shared_ptr<Value> f = x * q;
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
}
