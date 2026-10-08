# autograd-cpp

A small scalar autograd engine and neural net library in C++20, in the spirit of
[micrograd](https://github.com/karpathy/micrograd). Every number is a `Value` node in a
computation graph, and calling `backprop` on an output fills in the gradient of every node
that went into it.

Still a work in progress. Everything currently lives in [main.cpp](main.cpp). The files in
`src/`, `examples/` and `tests/` are placeholders for when it gets split into modules.

## What's there

**Engine**
- `Value`: a heap-allocated node (always used through `shared_ptr<Value>`) holding `data`, `grad`,
  its inputs, the op that produced it, and a `backward` closure that pushes its grad down to its inputs.
- Ops: `+`, `*`, `/` (each also works with a plain `double` on either side) and `tanh`.
- `backprop(output)`: topologically sorts the graph, seeds `output->grad = 1`, then runs every
  node's `backward` in reverse order. Grads are not zeroed first, so running it twice accumulates.
- `grad_check(f, x)`: compares backprop's gradient against a central-difference numerical
  gradient and returns the relative error.

**Neural net**
- `Neuron(n_inputs)`: random weights and bias in [-1, 1], computes `tanh(w·x + b)`.
- `Layer(n_neurons, n_inputs)`: a list of neurons that all see the same inputs.
- `MLP(n_inputs, layer_sizes)`: layers chained together. The last layer size should be 1,
  since the forward pass returns a single `Value`.
- Each of these has `parameters()`, which returns every weight and bias so you can update them.

## Example

```cpp
// gradient of a small expression
auto a = make_shared<Value>(2.0);
auto b = make_shared<Value>(-3.0);
auto c = tanh(a * b + 1);
backprop(c);
// a->grad and b->grad now hold dc/da and dc/db

// check it against the numerical gradient
double err = grad_check([](const shared_ptr<Value>& x) { return tanh(x * x / 3); }, 0.5);

// one forward/backward/update step on an MLP
MLP model(3, {4, 4, 1});
vector<shared_ptr<Value>> x = {make_shared<Value>(1.0), make_shared<Value>(-2.0), make_shared<Value>(0.5)};
auto diff = model(x) + -1.0;  // target 1.0 (no subtraction op yet)
auto loss = diff * diff;
backprop(loss);
for (auto& p : model.parameters()) {
    p->data += -0.05 * p->grad;
    p->grad = 0;
}
```

## Building

Requires CMake 3.20+ and a C++20 compiler.

```sh
cmake -S . -B build
cmake --build build
./build/main
```

`main()` is currently empty, so the binary doesn't do anything yet.

## TODO

- Ops: subtraction, negation, `exp`, `pow`
- Split `main.cpp` into `src/value.*` and `src/nn.*`
- Training example (`examples/xor.cpp`) and grad check tests (`tests/grad_check.cpp`)
