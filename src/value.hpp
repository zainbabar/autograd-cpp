#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

// TODO: add negation, exp

// which op produced a node (NONE for leaves)
enum class Op {NONE, ADD, SUB, MULT, DIV, POW, TANH};

// Every node lives on the heap behind a shared_ptr, so reusing a node in several ops means
// they all point at the same object and its grad accumulates in one place.
// Each node also holds shared_ptrs to its inputs, which keeps the whole graph alive
// for the backward pass as long as the output is alive.
class Value {
  public:
    double data;
    std::vector<std::shared_ptr<Value>> inputs;  // nodes this one was computed from (empty for leaves)
    double grad;
    Op op;
    // pushes this node's grad down into its inputs' grads, set by whichever op created the node
    std::function<void()> backward;

    Value(
        double data,
        std::vector<std::shared_ptr<Value>> inputs=std::vector<std::shared_ptr<Value>>{},
        Op op=Op::NONE
    ): data{data}, inputs{inputs}, grad{0}, op{op}, backward([]{}) {}
};

inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& self, const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(
        self->data + other->data,
        std::vector<std::shared_ptr<Value>>{self, other},
        Op::ADD
    );
    // The lambda captures a raw pointer to out, not a reference or a shared_ptr:
    // - self/other/out are locals, so capturing by reference would dangle once this returns
    // - a shared_ptr copy would make out own itself (ref cycle), so it would never be freed
    // The lambda is stored inside out itself, so the raw pointer is always valid whenever it runs.
    // d(a+b)/da = d(a+b)/db = 1, so each input just gets the incoming grad.
    // += since a node can feed several outputs, its grad is the sum over all of them (also handles a + a)
    out->backward = [o = out.get()] {
        for (auto& input : o->inputs) {
            input->grad += 1 * o->grad;
        }
    };
    return out;
}

// scalar overloads: wrap the double in a leaf Value and reuse the Value version
inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& self, double other) {
    return self + std::make_shared<Value>(other);
}

inline std::shared_ptr<Value> operator+(double other, const std::shared_ptr<Value>& self) {
    return self + other;
}

inline std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& self, const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(
        self->data - other->data,
        std::vector<std::shared_ptr<Value>>{self, other},
        Op::SUB
    );
    // d(a-b)/da = 1 and d(a-b)/db = -1
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += 1 * o->grad;
        o->inputs[1]->grad += -1 * o->grad;
    };
    return out;
}

inline std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& self, double other) {
    return self - std::make_shared<Value>(other);
}

inline std::shared_ptr<Value> operator-(double other, const std::shared_ptr<Value>& self) {
    return std::make_shared<Value>(other) - self;
}

inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& self, const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(
        self->data * other->data,
        std::vector<std::shared_ptr<Value>>{self, other},
        Op::MULT
    );
    // d(a*b)/da = b and d(a*b)/db = a, so each input's grad is the other input's value * incoming grad
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += o->inputs[1]->data * o->grad;
        o->inputs[1]->grad += o->inputs[0]->data * o->grad;
    };
    return out;
}

inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& self, double other) {
    return self * std::make_shared<Value>(other);
}

inline std::shared_ptr<Value> operator*(double other, const std::shared_ptr<Value>& self) {
    return self * other;
}

inline std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& self, const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = std::make_shared<Value>(
        self->data / other->data,
        std::vector<std::shared_ptr<Value>>{self, other},
        Op::DIV
    );
    // d(a/b)/da = 1/b and d(a/b)/db = -a/b^2
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += (1 / o->inputs[1]->data) * o->grad;
        o->inputs[1]->grad += -(o->inputs[0]->data / std::pow(o->inputs[1]->data, 2)) * o->grad;
    };
    return out;
}

inline std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& self, double other) {
    return self / std::make_shared<Value>(other);
}

inline std::shared_ptr<Value> operator/(double other, const std::shared_ptr<Value>& self) {
    return std::make_shared<Value>(other) / self;
}

// raises a Value to a constant power, the exponent is a plain double so it doesn't get a grad
inline std::shared_ptr<Value> pow(const std::shared_ptr<Value>& self, double n) {
    std::shared_ptr<Value> out = std::make_shared<Value>(
        std::pow(self->data, n),
        std::vector<std::shared_ptr<Value>>{self},
        Op::POW
    );
    // d(x^n)/dx = n * x^(n-1), n isn't stored on the node so the lambda captures it by value
    out->backward = [o = out.get(), n] {
        o->inputs[0]->grad += n * std::pow(o->inputs[0]->data, n - 1) * o->grad;
    };
    return out;
}

// tanh as our activation function for now
inline std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& self) {
    double x = self->data;
    // tanh(x) = (e^2x - 1) / (e^2x + 1), but for large positive x e^2x overflows to inf and gives inf/inf = NaN.
    // For x >= 0, use the same formula multiplied through by e^-2x so the exponent is never positive.
    double t = 0;
    if (x < 0) { t = (std::exp(2*x) - 1) / (std::exp(2*x) + 1); }
    else { t = (1 - std::exp(2*(-x))) / (1 + std::exp(2*(-x))); }
    std::shared_ptr<Value> out = std::make_shared<Value>(t, std::vector<std::shared_ptr<Value>>{self}, Op::TANH);
    // d tanh(x)/dx = 1 - tanh(x)^2, and tanh(x) is already stored in o->data
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += (1 - std::pow(o->data, 2)) * o->grad;
    };
    return out;
}
