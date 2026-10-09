#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>
#include "autograd.hpp"
#include "nn.hpp"
#include "data.hpp"
#include "train.hpp"
#include <cmath>
#include <numeric>    
#include <random>    
#include <algorithm>

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

// 1 training step for a batch
void train_step(MLP& model, shared_ptr<Value> batch_loss, double learning_rate) {
    zero_grad(model);
    backprop(batch_loss);
    for (auto& param : model.parameters()) {
        param->data -= param->grad * learning_rate;
    }
}

const int BATCH_SIZE = 32;

// prints accuracies the same way everywhere, including the FINAL line
string fmt_acc(double acc) {
    ostringstream out;
    out << fixed << setprecision(4) << acc;
    return out.str();
}

void train_epochs(MLP& model, const Dataset& train, const Dataset& test, const Dataset& train_eval,
                  int n_epochs, double learning_rate) {
    vector<int> indicies(train.labels.size()); // create vector w/ all 0s 
    iota(indicies.begin(), indicies.end(), 0); // fills inidices w consecutive values
    mt19937 rng(42);
    for (int epoch = 1; epoch <= n_epochs; ++epoch) {
        auto epoch_start = chrono::steady_clock::now();
        // shuffle the data for each epoch so batches r random
        double epoch_loss = 0;
        double batches = 0;
        shuffle(indicies.begin(), indicies.end(), rng);
        // 1 full pass of the training data, using minibatch gd
        for (size_t start = 0; start + BATCH_SIZE <= indicies.size(); start += BATCH_SIZE) {
            ++batches;
            // create a batch 
            vector<int> batch_indicies(indicies.begin() + start, indicies.begin() + start + BATCH_SIZE);
            auto loss = batch_loss(model, train, batch_indicies);
            // train step for this batch
            epoch_loss += loss->data;
            train_step(model, loss, learning_rate);
        }
        // avg epoch loss, test acc, train acc, and how long the epoch took (eval included)
        double avg_epoch_loss = epoch_loss / batches;
        auto test_acc = accuracy(model, test);
        auto train_acc = accuracy(model, train_eval);
        double secs = chrono::duration<double>(chrono::steady_clock::now() - epoch_start).count();
        cout << "epoch " << epoch << "  loss " << fixed << setprecision(6) << avg_epoch_loss
             << "  test " << fmt_acc(test_acc) << "  train " << fmt_acc(train_acc)
             << "  time " << setprecision(1) << secs << "s" << endl;
    }
}

// usage: ./build/mnist <train_rows> <epochs> <lr>, train_rows = 0 loads all 60k
int main(int argc, char** argv) {
    size_t train_rows = 5000;
    int epochs = 20;
    double lr = 0.1;
    try {
        if (argc > 1) { train_rows = stoul(argv[1]); }
        if (argc > 2) { epochs = stoi(argv[2]); }
        if (argc > 3) { lr = stod(argv[3]); }
    } catch (const exception&) {
        cerr << "usage: " << argv[0] << " <train_rows> <epochs> <lr>" << endl;
        return 1;
    }

    Dataset train = load_csv("data/mnist_train.csv", train_rows);
    Dataset test = load_csv("data/mnist_test.csv", 2000);
    // fixed set of the first 1000 train images to track train accuracy on
    Dataset train_eval = load_csv("data/mnist_train.csv", 1000);
    if (train.labels.size() < BATCH_SIZE) {
        cerr << "need at least " << BATCH_SIZE << " train rows" << endl;
        return 1;
    }

    MLP model(784, {16, 10});
    scale_init(model);

    // loss on the first batch before any training, so there's something to compare epoch losses against
    vector<int> first_batch(BATCH_SIZE);
    iota(first_batch.begin(), first_batch.end(), 0);
    double initial_loss = batch_loss(model, train, first_batch)->data;

    cout << "train_rows " << train.labels.size() << "  epochs " << epochs << "  lr " << lr
         << "  batch_size " << BATCH_SIZE << "  params " << model.parameters().size() << endl;
    cout << "initial loss " << fixed << setprecision(6) << initial_loss << endl;

    train_epochs(model, train, test, train_eval, epochs, lr);

    Dataset test_full = load_csv("data/mnist_test.csv");
    cout << "FINAL test10k " << fmt_acc(accuracy(model, test_full)) << endl;
}
