#include <iostream>
#include <memory>
#include <set>
#include <vector>
using namespace std;

// our ops for tracking backward pass
enum class Op {NONE, ADD, MULT};

class Value {
  public:
    vector<shared_ptr<Value>> inputs;  // the inputs that led to this current node
    double data; // data
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


Value operator+(Value& self, Value& other) { // create a new Value object via addition
    Value new_value = Value(
        (self.data + other.data), // sum (+), data in the new thing
        vector<shared_ptr<Value>>{
            make_shared<Value>(self), make_shared<Value>(other)},
            Op::ADD
    );
    return new_value;
}

Value operator+(Value& self, double other) { // incase we want to do scalar add
    Value vother = Value(other);
    return self + vother;
}

Value operator+(double other, Value& self) { // incase we want to do scalar add
    return self + other;
}

Value operator*(Value& self, Value& other) { // create a new value object via multiplicaiton
    Value new_value = Value(
        (self.data * other.data),
        vector<shared_ptr<Value>>{
        make_shared<Value>(self), make_shared<Value>(other)},
        Op::MULT
     );
    return new_value;
}

Value operator*(Value& self, double other) { // incase we want to do scalar mult
    Value vother = Value(other);
    return self * vother;
}

Value operator*(double other, Value& self) { // incase we want to do scalar add
    return self * other;
}

// have working both scalar add mult and and value add mult 
// now need to write a backward function to comptue its gradient


int main() {
    cout << "hello world" << endl;
    Value x = Value(2);
    Value y = Value(3);
    cout << x.data << ' ' << y.data << endl;
    Value q = x + y;
    Value z = q * 2.0;
    cout << z.data << endl;
    for (auto v : z.inputs) {
        cout << v->data << endl;
    }
}
