#include <iostream>
#include <memory>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <functional>
#include <random>
#include <string>
#include <cassert>
#include <iomanip>
using namespace std;


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
shared_ptr<Value> operator+(const shared_ptr<Value>& self, double other) {
    return self + make_shared<Value>(other);
}

shared_ptr<Value> operator+(double other, const shared_ptr<Value>& self) {
    return self + other;
}

shared_ptr<Value> operator-(const shared_ptr<Value>& self, const shared_ptr<Value>& other) {
    shared_ptr<Value> out = make_shared<Value>(
        self->data - other->data,
        vector<shared_ptr<Value>>{self, other},
        Op::SUB
    );
    // d(a-b)/da = 1 and d(a-b)/db = -1
    out->backward = [o = out.get()] {
        o->inputs[0]->grad += 1 * o->grad;
        o->inputs[1]->grad += -1 * o->grad;
    };
    return out;
}

shared_ptr<Value> operator-(const shared_ptr<Value>& self, double other) {
    return self - make_shared<Value>(other);
}

shared_ptr<Value> operator-(double other, const shared_ptr<Value>& self) {
    return make_shared<Value>(other) - self;
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

// raises a Value to a constant power, the exponent is a plain double so it doesn't get a grad
shared_ptr<Value> pow(const shared_ptr<Value>& self, double n) {
    shared_ptr<Value> out = make_shared<Value>(
        pow(self->data, n),
        vector<shared_ptr<Value>>{self},
        Op::POW
    );
    // d(x^n)/dx = n * x^(n-1), n isn't stored on the node so the lambda captures it by value
    out->backward = [o = out.get(), n] {
        o->inputs[0]->grad += n * pow(o->inputs[0]->data, n - 1) * o->grad;
    };
    return out;
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

    // weights + bias, so a training loop can update each one by lr * grad after backprop
    vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> params = weights;
        params.push_back(bias);
        return params;
    }
};

class Layer {
  public:
    unsigned long n_neurons;
    unsigned long n_inputs;
    vector<Neuron> neurons;
    Layer(unsigned long n_neurons, unsigned long n_inputs): 
        n_neurons{n_neurons}, n_inputs{n_inputs} {
        // every neuron in the layer sees the same n_inputs
        for (int i = 0; i < n_neurons; ++i) {
            neurons.push_back(Neuron{n_inputs});
        }
    }
    // runs the inputs through every neuron, returns one output per neuron
    vector<shared_ptr<Value>> operator()(const vector<shared_ptr<Value>>& inputs) {
        vector<shared_ptr<Value>> outputs{};
        for (auto& neuron : neurons) {
            outputs.push_back(neuron(inputs));
        }
        return outputs;
    }

    // every neuron's params concatenated into one list
    vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> params{};
        for (auto& neuron : neurons) {
            vector<shared_ptr<Value>> nparams = neuron.parameters();
            params.insert(params.end(), nparams.begin(), nparams.end());
        }
        return params;
    }
};

// The last entry in layer_sizes should be 1: operator() returns a single Value,
// so it only takes the first output of the last layer and drops the rest.
class MLP {
  public:
    unsigned long n_inputs;
    unsigned long n_layers;
    vector<unsigned long> layer_sizes;
    vector<Layer> layers;
    // layer_sizes[i] is the number of neurons in layer i
    MLP(unsigned long n_inputs, vector<unsigned long> layer_sizes):
        n_inputs{n_inputs}, n_layers{layer_sizes.size()}, layer_sizes{layer_sizes} {
        // the first layer takes the network's inputs, every later layer takes the previous layer's outputs
        layers.push_back(Layer{layer_sizes[0], n_inputs});
        for (int i = 1; i < n_layers; ++i) {
            layers.push_back(Layer{layer_sizes[i], layer_sizes[i - 1]});
        }
    }
    // forward pass: feeds each layer's outputs into the next and returns the final output
    shared_ptr<Value> operator()(const vector<shared_ptr<Value>>& inputs) {
        assert(inputs.size() == n_inputs);
        vector<shared_ptr<Value>> outputs = inputs;
        for (auto& layer : layers) {
            outputs = layer(outputs);
        }
        shared_ptr<Value> output = outputs[0];
        return output;
    }
    // every layer's params concatenated into one list
    vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> params{};
        for (auto& layer : layers) {
            vector<shared_ptr<Value>> lparams = layer.parameters();
            params.insert(params.end(), lparams.begin(), lparams.end());
        }
        return params;
    }
};

// wrap one row of plain numbers in fresh leaf nodes for a forward pass
vector<shared_ptr<Value>> to_nodes(const vector<double>& row) {
    vector<shared_ptr<Value>> nodes;
    for (double d : row) {
        nodes.push_back(make_shared<Value>(d));
    }
    return nodes;
}

// sum of squared errors over the whole dataset.
// it's all one graph, so a single backprop on the result gives every param its grad for the full batch
shared_ptr<Value> loss(MLP& model, const vector<vector<double>>& data, const vector<double>& targets) {
    shared_ptr<Value> total = make_shared<Value>(0);
    for (size_t i = 0; i < data.size(); ++i) {
        shared_ptr<Value> pred = model(to_nodes(data[i]));
        total = total + pow(pred - targets[i], 2.0);
    }
    return total;
}

// backprop accumulates into grad, so it has to be reset every step or old grads leak into the update
void zero_grad(MLP& model) {
    for (auto& param : model.parameters()) {
        param->grad = 0;
    }
}

// plain gradient descent: forward, zero grads, backward, then nudge every param against its grad
void train(MLP& model, const vector<vector<double>>& data, const vector<double>& targets,
           int epochs, double lr, int log_every) {
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        shared_ptr<Value> L = loss(model, data, targets);
        if (epoch % log_every == 0) {
            cout << "  epoch " << setw(4) << epoch << "  loss " << L->data << endl;
        }
        zero_grad(model);
        backprop(L);
        for (auto& param : model.parameters()) {
            param->data -= lr * param->grad;
        }
    }
}

// prints each prediction next to its target, then how many the model got right.
// a prediction counts as right if it has the same sign as the target, since targets are -1/1
void print_predictions(MLP& model, const vector<vector<double>>& inputs, const vector<double>& targets) {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        shared_ptr<Value> pred = model(to_nodes(inputs[i]));
        bool right = (pred->data > 0) == (targets[i] > 0);
        if (right) { ++correct; }
        cout << "  (" << int(inputs[i][0]) << ", " << int(inputs[i][1]) << ")"
             << "  target " << setw(2) << int(targets[i])
             << "  pred " << showpos << pred->data << noshowpos
             << (right ? "  ok" : "  WRONG") << endl;
    }
    cout << "\n  " << correct << "/" << inputs.size() << " correct"
         << (correct == int(inputs.size()) ? ", converged" : ", didn't converge") << endl;
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

    // 3. train an MLP on XOR, which a single neuron can't learn since it isn't linearly separable.
    // targets are -1/1 instead of 0/1 to match tanh's output range
    cout << "\n== training a 2-4-3-1 MLP on XOR ==" << endl;
    const vector<vector<double>> xor_inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const vector<double> xor_targets = {-1, 1, 1, -1};
    MLP model(2, {4, 3, 1});
    cout << "  " << model.parameters().size() << " parameters" << endl;
    train(model, xor_inputs, xor_targets, 3000, 0.01, 500);

    cout << "\n== predictions ==" << endl;
    print_predictions(model, xor_inputs, xor_targets);
}
