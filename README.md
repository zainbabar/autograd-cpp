# autograd-cpp

A small scalar autograd engine and neural net library in C++20, in the spirit of
[micrograd](https://github.com/karpathy/micrograd). Every number is a `Value` node in a
computation graph, and calling `backprop` on an output fills in the gradient of every node
that went into it. On top of that there's a tiny MLP that can be trained with plain gradient descent.
It trains on XOR and MNIST.

The library is header-only, in `src/`:
- [value.hpp](src/value.hpp): `Value`, the ops, `pow` and `tanh`
- [autograd.hpp](src/autograd.hpp): topological sort (`build`) and `backprop`
- [nn.hpp](src/nn.hpp): `Neuron`, `Layer`, `MLP`
- [train.hpp](src/train.hpp): `to_nodes`, `loss`, `zero_grad`, `train`
- [data.hpp](src/data.hpp): `Dataset`, `load_csv`, `print_digit`

[tests/grad_check.cpp](tests/grad_check.cpp) checks the gradients, [examples/xor.cpp](examples/xor.cpp) trains an MLP on XOR,
and [examples/mnist.cpp](examples/mnist.cpp) trains a classifier on MNIST.

## Building and running

Requires CMake 3.20+ and a C++20 compiler.

```sh
cmake -S . -B build
cmake --build build
./build/grad_check
./build/xor
./build/mnist
```

The build defaults to `Release`, since scalar autograd on MNIST is painfully slow without optimizations.

`mnist` needs the dataset as CSVs in `data/` (gitignored). [load_mnist.py](load_mnist.py) downloads it
from OpenML with scikit-learn and writes `data/mnist_train.csv` (60k rows) and `data/mnist_test.csv` (10k rows),
one image per row as `label,pixel0,...,pixel783`:

```sh
pip install scikit-learn
python load_mnist.py
```

`grad_check` runs backprop through a small expression and grad checks every op. `xor` trains
an MLP on XOR. Output looks something like this (weights are randomly initialised so the
training numbers change each run):

```
== gradients of c = tanh(a*b + a/b) ==
  c = -0.794432
  dc/da = -0.799235
  dc/db = 0.102466

== grad check (relative error vs numerical) ==
  x + 3             1.10e-13
  2 - x             1.10e-13
  x * x             1.89e-13
  1 / x             2.04e-08
  x^3               6.80e-09
  tanh(x)           3.20e-10
  tanh(x*x/3) - x   1.46e-09
```

```
== training a 2-4-3-1 MLP on XOR ==
  31 parameters
  epoch    0  loss 4.660180
  epoch  500  loss 0.081877
  epoch 1000  loss 0.017401
  ...
  epoch 3000  loss 0.003232

== predictions ==
  (0, 0)  target -1  pred -0.980843  ok
  (0, 1)  target  1  pred +0.979051  ok
  (1, 0)  target  1  pred +0.978775  ok
  (1, 1)  target -1  pred -0.978013  ok

  4/4 correct, converged
```

## What's there

**Engine**
- `Value`: a heap-allocated node (always used through `shared_ptr<Value>`) holding `data`, `grad`,
  its inputs, the op that produced it, and a `backward` closure that pushes its grad down to its inputs.
- Ops: `+`, `-`, `*`, `/` (each also works with a plain `double` on either side), `pow(x, n)` with a
  constant exponent, and `tanh`.
- `backprop(output)`: topologically sorts the graph, seeds `output->grad = 1`, then runs every
  node's `backward` in reverse order. Grads are not zeroed first, so running it twice accumulates.
- `grad_check(f, x)` (in `tests/grad_check.cpp`): compares backprop's gradient against a central-difference numerical
  gradient and returns the relative error.

**Neural net**
- `Neuron(n_inputs)`: random weights and bias in [-1, 1], computes `tanh(w·x + b)`.
- `Layer(n_neurons, n_inputs)`: a list of neurons that all see the same inputs.
- `MLP(n_inputs, layer_sizes)`: layers chained together. The last layer size should be 1,
  since the forward pass returns a single `Value`.
- Each of these has `parameters()`, which returns every weight and bias so you can update them.

**Training**
- `loss(model, data, targets)`: sum of squared errors over the whole dataset, as one graph.
- `zero_grad(model)`: resets every parameter's grad to 0.
- `train(model, data, targets, epochs, lr, log_every)`: full-batch gradient descent, prints the loss every `log_every` epochs.
- `to_nodes(row)`: wraps a row of plain `double`s in fresh leaf `Value`s for a forward pass.

**Data**
- `load_csv(path, max_rows)`: reads a `label,pixels...` CSV into a `Dataset` (`pixels` and `labels`, matched by index).
  Pixels are scaled from 0-255 to [0, 1], a header row is skipped, and `max_rows = 0` reads everything.
- `print_digit(pixels)`: draws a 28x28 image in the terminal as ASCII, handy for checking an image matches its label.

## MNIST

[examples/mnist.cpp](examples/mnist.cpp) trains a 784-16-10 MLP (tanh everywhere, ±1 targets, squared error)
with minibatch gradient descent, batch size 32.

**1. Export the data** (writes `data/mnist_train.csv` and `data/mnist_test.csv`, both gitignored):

```sh
pip install scikit-learn
python load_mnist.py
```

**2. Build in Release mode:**

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**3. Run** from the repo root. The binary must be run from there, since it opens `data/...` relative to the working directory:

```sh
./build/mnist <train_rows> <epochs> <lr>   # defaults: 5000 20 0.1, train_rows = 0 loads all 60,000
```

It prints the config and parameter count, then one line per epoch with the average batch loss,
accuracy on a 2,000-image test subset, accuracy on the first 1,000 train images, and the epoch time.
After the last epoch it evaluates the full 10k test set and prints `FINAL test10k <accuracy>`.

[scripts/run_seeds.sh](scripts/run_seeds.sh) runs 5 copies with the default config in parallel
(logs in `logs/seed_<i>.txt`) and prints the mean and sample std of their `FINAL test10k` values.
Weight init is random per run; the shuffle seed is fixed.

**Results**

| config | FINAL test10k (mean ± std, 5 runs) | time per epoch |
|---|---|---|
| default (5000 20 0.1) | TODO | TODO |
| full train set (0 20 0.1) | TODO | TODO |

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

// one forward/backward/update step on an MLP by hand
MLP model(3, {4, 4, 1});
vector<shared_ptr<Value>> x = {make_shared<Value>(1.0), make_shared<Value>(-2.0), make_shared<Value>(0.5)};
auto loss = pow(model(x) - 1.0, 2);  // target 1.0
backprop(loss);
for (auto& p : model.parameters()) {
    p->data -= 0.05 * p->grad;
    p->grad = 0;
}
```

## TODO

- MNIST: fill in the results table
- Ops: negation, `exp`, more activations (ReLU)
