#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <random>
#include <vector>

#include "value.hpp"

class Neuron {
  public:
    unsigned long n_inputs;
    // weights and bias are Values so backprop gives them grads we can use to update them in training
    std::vector<std::shared_ptr<Value>> weights;
    std::shared_ptr<Value> bias;

    Neuron(unsigned long n): n_inputs{n} {
        // seeded from random_device, swap rd() for a fixed number to get reproducible runs
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1, 1);
        for (int i = 0; i < n_inputs; ++i) {
            weights.push_back(std::make_shared<Value>(dist(gen)));
        }
        bias = std::make_shared<Value>(dist(gen));
    }
    // returns tanh(w·x + b) as a new Value
    // if inputs and weights differ in length, only the first min of the two are used
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
        std::shared_ptr<Value> dot = std::make_shared<Value>(0);
        for (int i = 0; i < std::min(inputs.size(), n_inputs); ++i) {
            dot = dot + (weights[i] * inputs[i]);
        }
        dot = dot + bias;
        std::shared_ptr<Value> output = tanh(dot);
        return output;
    }

    // weights + bias, so a training loop can update each one by lr * grad after backprop
    std::vector<std::shared_ptr<Value>> parameters() {
        std::vector<std::shared_ptr<Value>> params = weights;
        params.push_back(bias);
        return params;
    }
};

class Layer {
  public:
    unsigned long n_neurons;
    unsigned long n_inputs;
    std::vector<Neuron> neurons;
    Layer(unsigned long n_neurons, unsigned long n_inputs): 
        n_neurons{n_neurons}, n_inputs{n_inputs} {
        // every neuron in the layer sees the same n_inputs
        for (int i = 0; i < n_neurons; ++i) {
            neurons.push_back(Neuron{n_inputs});
        }
    }
    // runs the inputs through every neuron, returns one output per neuron
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
        std::vector<std::shared_ptr<Value>> outputs{};
        for (auto& neuron : neurons) {
            outputs.push_back(neuron(inputs));
        }
        return outputs;
    }

    // every neuron's params concatenated into one list
    std::vector<std::shared_ptr<Value>> parameters() {
        std::vector<std::shared_ptr<Value>> params{};
        for (auto& neuron : neurons) {
            std::vector<std::shared_ptr<Value>> nparams = neuron.parameters();
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
    std::vector<unsigned long> layer_sizes;
    std::vector<Layer> layers;
    // layer_sizes[i] is the number of neurons in layer i
    MLP(unsigned long n_inputs, std::vector<unsigned long> layer_sizes):
        n_inputs{n_inputs}, n_layers{layer_sizes.size()}, layer_sizes{layer_sizes} {
        // the first layer takes the network's inputs, every later layer takes the previous layer's outputs
        layers.push_back(Layer{layer_sizes[0], n_inputs});
        for (int i = 1; i < n_layers; ++i) {
            layers.push_back(Layer{layer_sizes[i], layer_sizes[i - 1]});
        }
    }
    // forward pass: feeds each layer's outputs into the next and returns the final output
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
        assert(inputs.size() == n_inputs);
        std::vector<std::shared_ptr<Value>> outputs = inputs;
        for (auto& layer : layers) {
            outputs = layer(outputs);
        }
        std::shared_ptr<Value> output = outputs[0];
        return output;
    }
    // every layer's params concatenated into one list
    std::vector<std::shared_ptr<Value>> parameters() {
        std::vector<std::shared_ptr<Value>> params{};
        for (auto& layer : layers) {
            std::vector<std::shared_ptr<Value>> lparams = layer.parameters();
            params.insert(params.end(), lparams.begin(), lparams.end());
        }
        return params;
    }
};
