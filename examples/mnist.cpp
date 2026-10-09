#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>
#include "autograd.hpp"
#include "nn.hpp"
#include "data.hpp"
#include "train.hpp"
#include <cmath>

using namespace std;

// scale weights of model since pre activated weighted sum would 
// have huge variance, tanh would be stuck at +-1
void scale_init(MLP& model) {
    // scale every weight by 1/sqrt(n) for each layer, cancels growth
    for (auto& layer : model.layers) {
        double scale = 1 / sqrt(layer.n_inputs);
        for (auto& neuron : layer.neurons) {
            for (auto& weight : neuron.weights) {
                weight->data *= scale;
            }
            neuron.bias->data *= scale;
        }
    }
}

auto layer_max = [](Layer& layer) {
    double mx = 0;
    for (auto& n : layer.neurons) {
        for (auto& w : n.weights) { mx = max(mx, abs(w->data)); }
        mx = max(mx, abs(n.bias->data));
    }
    return mx;
};

// same as operator() for mlp, just w/ no unwrap, so returns the vector 
// of logits for all class preds not just 1 like for xor
vector<shared_ptr<Value>> forward_all(MLP& model, const vector<shared_ptr<Value>>& inputs) {
    assert(inputs.size() == model.n_inputs);
    auto outputs = inputs;
    for (auto& layer : model.layers) {
        outputs = layer(outputs);
    }
    return outputs;
}

// build a target label since labels r just ints rn
// -1 for everything except the target, 1 for target
vector<double> target_vector(int label) {
    vector<double> tv{};
    for (int i = 0; i < 10; ++i) {
        if (i == label) { tv.push_back(1); }
        else { tv.push_back(-1); }
    }
    return tv;
}

// accuracy function
// for each image in dataset do forwward pass, largest logit is pred
// count % where it maches the actual label
double accuracy(MLP& model, const Dataset& data) {
    // how many instances we have in the dataset
    // pixles and labels are matched up by indicies
    int n = data.pixels.size();
    const auto& images = data.pixels;
    const auto& labels = data.labels;
    double correct = 0;
    for (int i = 0; i < n; ++i) {
        // turn current image into vector of Values for forward pass
        auto inputs = to_nodes(images[i]);
        auto outputs = forward_all(model, inputs);
        // find what the model predicted 
        int pred = 0;
        for (int j = 0; j < outputs.size(); ++j) {
            if (outputs[j]->data > outputs[pred]->data)  {
                pred = j;
            }
        }
        // see if its right, add to running total of right
        if (pred == labels[i]) {
            ++correct;
        }
        
    }
    return (correct / n);
}

// takes indicies of batch, returns a VALUE batch loss so we can backprop
shared_ptr<Value> batch_loss(MLP& model, const Dataset& data, vector<int> indicies) {
    shared_ptr<Value> total = make_shared<Value>(0); // value, keeps trakc of nodes
    for (int i : indicies) {
        auto preds = forward_all(model, to_nodes(data.pixels[i]));
        auto targs = target_vector(data.labels[i]);
        // preds and targets both vector 10 long, so we can compute an actual loss now 
        for (int j = 0; j < preds.size(); ++j) {
            // squared error for each output  
            total = total + pow((preds[j] - targs[j]), 2.0);
        }
    }
    auto avg_batch_loss = total / indicies.size();
    return avg_batch_loss;
}

void train_step(MLP& model, shared_ptr<Value> batch_loss, double learning_rate) {
    zero_grad(model);
    backprop(batch_loss);
    for (auto& param : model.parameters()) {
        param->data -= param->grad * learning_rate;
    }
}

void train_epochs(MLP& model, const Dataset& data, int epochs, double learning_rate) {

}



int main() {
    Dataset train = load_csv("data/mnist_train.csv", 5000);
    Dataset test = load_csv("data/mnist_test.csv", 2000);
    cout << "train: " << train.labels.size() << " rows, " << train.pixels[0].size() << " pixels each\n";
    cout << "test:  " << test.labels.size() << " rows\n";

    // vector<int> counts(10, 0);
    // for (int y : train.labels) { ++counts[y]; }
    // for (int d = 0; d < 10; ++d) { cout << "digit " << d << ": " << counts[d] << "\n"; }

    // cout << "first train image, label " << train.labels[0] << ":\n";
    // print_digit(train.pixels[0]);

    MLP model(784, {16, 10});
    // cout << layer_max(model.layers[0]) << endl;
    // cout << layer_max(model.layers[1]) << endl;
    scale_init(model);
    // cout << layer_max(model.layers[0]) << endl;
    // cout << layer_max(model.layers[1]) << endl;

    auto inputs = to_nodes(train.pixels[0]);
    auto outputs = forward_all(model, inputs);
    for (auto& out : outputs) {
        cout << out->data << " ";
    }
    cout << endl;
    cout << accuracy(model, train) << endl;
}