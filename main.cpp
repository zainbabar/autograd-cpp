#include <iostream>
#include <memory>
#include <vector>
using namespace std;

// our ops for tracking backward pass
enum class Op {NONE, ADD, MULT};

class Value {
  public:
    double data; // data
    vector<shared_ptr<Value>> inputs;  // the inputs that led to this current node
    double grad; // gradient
    Op op;

    Value(   
        double data, 
        vector<shared_ptr<Value>> inputs=vector<shared_ptr<Value>>{}, // empty by def 
        Op op=Op::NONE  // none by def
    ): data{data}, inputs{inputs}, grad{0}, op{op} {} 

    void backward() {
    }
};
shared_ptr<Value> operator+(const shared_ptr<Value>& self, const shared_ptr<Value>& other) {
    return make_shared<Value>(
        self->data + other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::ADD
    );
}

shared_ptr<Value> operator+(const shared_ptr<Value>& self, double other) {
    return self + make_shared<Value>(other);
}

shared_ptr<Value> operator+(double other, const shared_ptr<Value>& self) {
    return self + other;
}

shared_ptr<Value> operator*(const shared_ptr<Value>& self, const shared_ptr<Value>& other) {
    return make_shared<Value>(
        self->data * other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::MULT
    );
}

shared_ptr<Value> operator*(const shared_ptr<Value>& self, double other) {
    return self * make_shared<Value>(other);
}

shared_ptr<Value> operator*(double other, const shared_ptr<Value>& self) {
    return self * other;
}
// have working both scalar add mult and and value add mult 
// now need to write a backward function to comptue its gradient
// ownership issue, need to make nodes live on the heap from the start
// so the operators deal with pointers

int main() {
    shared_ptr<Value> x = make_shared<Value>(5);
    shared_ptr<Value> y = make_shared<Value>(3);
    shared_ptr<Value> z = x + y;
}
