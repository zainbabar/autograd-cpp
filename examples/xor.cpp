#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "nn.hpp"
#include "train.hpp"
#include "value.hpp"
using namespace std;

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
