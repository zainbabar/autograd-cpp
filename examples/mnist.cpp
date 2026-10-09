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

// Neuron inits weights and bias uniform in [-1, 1], which is fine for xor's 2 inputs but not for 784.
// the pre activation sum adds up 784 random terms, so its variance grows w/ the number of inputs
// and tanh gets pushed way out to +-1 where its grad is basically 0, so nothing learns.
// scaling by 1/sqrt(n) cancels that growth (variance of the sum scales w/ n, so std scales w/ sqrt(n)).
// done here instead of in Neuron so xor's init stays the same
void scale_init(MLP& model) {
    for (auto& layer : model.layers) {
        // n is per layer: 784 for the hidden layer, 16 for the output layer
        double scale = 1 / sqrt(layer.n_inputs);
        for (auto& neuron : layer.neurons) {
            for (auto& weight : neuron.weights) {
                weight->data *= scale;
            }
            // bias gets scaled too, a [-1, 1] bias alone can already push tanh far from 0
            neuron.bias->data *= scale;
        }
    }
}

// debug helper: biggest |weight| or |bias| in a layer, used to check scale_init actually shrank things
auto layer_max = [](Layer& layer) {
    double mx = 0;
    for (auto& n : layer.neurons) {
        for (auto& w : n.weights) { mx = max(mx, abs(w->data)); }
        mx = max(mx, abs(n.bias->data));
    }
    return mx;
};

// same as MLP::operator() but w/o the unwrap at the end. operator() only returns outputs[0]
// since xor has 1 output, here we need all 10 (one score per digit).
// written here instead of changing MLP so xor keeps working as is
vector<shared_ptr<Value>> forward_all(MLP& model, const vector<shared_ptr<Value>>& inputs) {
    assert(inputs.size() == model.n_inputs);
    auto outputs = inputs;
    for (auto& layer : model.layers) {
        outputs = layer(outputs);
    }
    return outputs;
}

// labels r just ints 0-9, but the loss needs a target for each of the 10 outputs.
// one hot style but w/ -1 instead of 0 since every output goes through tanh, which lives in (-1, 1),
// so 1 for the true digit and -1 for the other 9
vector<double> target_vector(int label) {
    vector<double> tv{};
    for (int i = 0; i < 10; ++i) {
        if (i == label) { tv.push_back(1); }
        else { tv.push_back(-1); }
    }
    return tv;
}

// fraction of images (0 to 1, not a %) where the model's pred matches the label.
// the pred is whichever of the 10 outputs is biggest (argmax).
// no backprop here, but forward_all still builds a graph per image, it just gets thrown away
double accuracy(MLP& model, const Dataset& data) {
    // pixels[i] and labels[i] are the same image, matched up by index
    int n = data.pixels.size();
    const auto& images = data.pixels;
    const auto& labels = data.labels;
    double correct = 0;
    for (int i = 0; i < n; ++i) {
        auto inputs = to_nodes(images[i]);
        auto outputs = forward_all(model, inputs);
        // argmax: index of the biggest output, ties go to the lower digit
        int pred = 0;
        for (int j = 0; j < outputs.size(); ++j) {
            if (outputs[j]->data > outputs[pred]->data)  {
                pred = j;
            }
        }
        if (pred == labels[i]) {
            ++correct;
        }

    }
    return (correct / n);
}

// loss for one minibatch, returned as a Value (not a double) so it's the root of the graph and we can backprop it.
// squared error over all 10 outputs of every image, summed, then divided by the batch size.
// averaging over the batch (not summing) keeps the grad size the same no matter the batch size,
// so lr doesn't need retuning if the batch size changes. not divided by 10 though, so it's per image not per output
shared_ptr<Value> batch_loss(MLP& model, const Dataset& data, vector<int> indicies) {
    // starts as a 0 leaf, every image's errors get chained onto it so the whole batch is one graph
    shared_ptr<Value> total = make_shared<Value>(0);
    for (int i : indicies) {
        auto preds = forward_all(model, to_nodes(data.pixels[i]));
        auto targs = target_vector(data.labels[i]);
        // preds and targs r both 10 long, one squared error per output
        for (int j = 0; j < preds.size(); ++j) {
            total = total + pow((preds[j] - targs[j]), 2.0);
        }
    }
    auto avg_batch_loss = total / indicies.size();
    return avg_batch_loss;
}

// 1 step of gradient descent on a batch.
// zero first since backprop adds onto grad, otherwise last batch's grads leak into this update
void train_step(MLP& model, shared_ptr<Value> batch_loss, double learning_rate) {
    zero_grad(model);
    backprop(batch_loss);
    // step every param against its grad, since grad points in the direction that increases the loss
    for (auto& param : model.parameters()) {
        param->data -= param->grad * learning_rate;
    }
}

// 32 is a standard minibatch size: small enough that each step is cheap and we get lots of updates
// per epoch, big enough that the grad isn't super noisy
const int BATCH_SIZE = 32;

// accuracy as a string w/ 4 decimals, so the per epoch lines and the FINAL line always match
// (run_seeds.sh parses the FINAL one). uses its own stream so it doesn't mess w/ cout's formatting
string fmt_acc(double acc) {
    ostringstream out;
    out << fixed << setprecision(4) << acc;
    return out.str();
}

// minibatch gd: each epoch shuffles the train set, splits it into batches, and does a train step per batch.
// after each epoch prints avg loss, test acc (2k subset), train acc (first 1k train images), and the epoch time
void train_epochs(MLP& model, const Dataset& train, const Dataset& test, const Dataset& train_eval,
                  int n_epochs, double learning_rate) {
    // shuffle a list of indices instead of the dataset itself, so we never copy the images around
    vector<int> indicies(train.labels.size());
    iota(indicies.begin(), indicies.end(), 0); // 0, 1, 2, ..., n-1
    // fixed seed so the batch order is the same every run, only the weight init changes between runs
    mt19937 rng(42);
    for (int epoch = 1; epoch <= n_epochs; ++epoch) {
        auto epoch_start = chrono::steady_clock::now();
        double epoch_loss = 0;
        double batches = 0;
        // reshuffle every epoch so the model doesn't see the same batches in the same order each time
        shuffle(indicies.begin(), indicies.end(), rng);
        // start + BATCH_SIZE <= size drops the last partial batch, so every batch is a full 32.
        // that's at most 31 images skipped per epoch, and they're different ones each time since we shuffle
        for (size_t start = 0; start + BATCH_SIZE <= indicies.size(); start += BATCH_SIZE) {
            ++batches;
            vector<int> batch_indicies(indicies.begin() + start, indicies.begin() + start + BATCH_SIZE);
            auto loss = batch_loss(model, train, batch_indicies);
            // grab the loss before the step, so it's the loss the batch had going in
            epoch_loss += loss->data;
            train_step(model, loss, learning_rate);
        }
        // the epoch loss is the avg of the batch losses, and those were measured while the weights were still
        // changing, so it lags a bit behind where the model actually is by the end of the epoch
        double avg_epoch_loss = epoch_loss / batches;
        auto test_acc = accuracy(model, test);
        auto train_acc = accuracy(model, train_eval);
        // the time includes the two accuracy passes, not just training
        double secs = chrono::duration<double>(chrono::steady_clock::now() - epoch_start).count();
        // endl (not "\n") flushes every line, so you see progress live even when it's piped into a log file
        cout << "epoch " << epoch << "  loss " << fixed << setprecision(6) << avg_epoch_loss
             << "  test " << fmt_acc(test_acc) << "  train " << fmt_acc(train_acc)
             << "  time " << setprecision(1) << secs << "s" << endl;
    }
}

// usage: ./build/mnist <train_rows> <epochs> <lr>, train_rows = 0 loads all 60k.
// has to be run from the repo root, the csv paths are relative to the working directory
int main(int argc, char** argv) {
    // defaults if args r left out. 5000 rows instead of all 60k since scalar autograd is slow
    size_t train_rows = 5000;
    int epochs = 20;
    double lr = 0.1;
    // stoul/stoi/stod throw on junk like "abc", so catch it and print usage instead of crashing
    try {
        if (argc > 1) { train_rows = stoul(argv[1]); }
        if (argc > 2) { epochs = stoi(argv[2]); }
        if (argc > 3) { lr = stod(argv[3]); }
    } catch (const exception&) {
        cerr << "usage: " << argv[0] << " <train_rows> <epochs> <lr>" << endl;
        return 1;
    }

    Dataset train = load_csv("data/mnist_train.csv", train_rows);
    // per epoch test acc only uses 2k of the 10k test images to keep each epoch fast,
    // the full 10k only gets used once at the very end
    Dataset test = load_csv("data/mnist_test.csv", 2000);
    // train acc is always on the first 1000 rows of the train csv, so it's the same set every epoch.
    // loaded separately from train, so if train_rows < 1000 some of these were never trained on
    Dataset train_eval = load_csv("data/mnist_train.csv", 1000);
    // need at least one full batch, otherwise the epoch loop never runs and avg loss would be 0/0
    if (train.labels.size() < BATCH_SIZE) {
        cerr << "need at least " << BATCH_SIZE << " train rows" << endl;
        return 1;
    }

    // 784 pixel inputs -> 16 hidden -> 10 outputs, one per digit. small since every weight is its own node
    MLP model(784, {16, 10});
    scale_init(model);

    // loss on the first batch before any training, a baseline to compare the epoch losses against.
    // just measured, never backpropped, so the weights don't change
    vector<int> first_batch(BATCH_SIZE);
    iota(first_batch.begin(), first_batch.end(), 0);
    double initial_loss = batch_loss(model, train, first_batch)->data;

    cout << "train_rows " << train.labels.size() << "  epochs " << epochs << "  lr " << lr
         << "  batch_size " << BATCH_SIZE << "  params " << model.parameters().size() << endl;
    cout << "initial loss " << fixed << setprecision(6) << initial_loss << endl;

    train_epochs(model, train, test, train_eval, epochs, lr);

    // final score on the whole 10k test set, run_seeds.sh greps for this exact "FINAL test10k" line
    Dataset test_full = load_csv("data/mnist_test.csv");
    cout << "FINAL test10k " << fmt_acc(accuracy(model, test_full)) << endl;
}
