#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "autograd.hpp"
#include "nn.hpp"
#include "value.hpp"

// wrap one row of plain numbers in fresh leaf nodes for a forward pass
inline std::vector<std::shared_ptr<Value>> to_nodes(const std::vector<double>& row) {
    std::vector<std::shared_ptr<Value>> nodes;
    for (double d : row) {
        nodes.push_back(std::make_shared<Value>(d));
    }
    return nodes;
}

// sum of squared errors over the whole dataset.
// it's all one graph, so a single backprop on the result gives every param its grad for the full batch
inline std::shared_ptr<Value> loss(MLP& model, const std::vector<std::vector<double>>& data, const std::vector<double>& targets) {
    std::shared_ptr<Value> total = std::make_shared<Value>(0);
    for (size_t i = 0; i < data.size(); ++i) {
        std::shared_ptr<Value> pred = model(to_nodes(data[i]));
        total = total + pow(pred - targets[i], 2.0);
    }
    return total;
}

// backprop accumulates into grad, so it has to be reset every step or old grads leak into the update
inline void zero_grad(MLP& model) {
    for (auto& param : model.parameters()) {
        param->grad = 0;
    }
}

// plain gradient descent: forward, zero grads, backward, then nudge every param against its grad
inline void train(MLP& model, const std::vector<std::vector<double>>& data, const std::vector<double>& targets,
           int epochs, double lr, int log_every) {
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        std::shared_ptr<Value> L = loss(model, data, targets);
        if (epoch % log_every == 0) {
            std::cout << "  epoch " << std::setw(4) << epoch << "  loss " << L->data << std::endl;
        }
        zero_grad(model);
        backprop(L);
        for (auto& param : model.parameters()) {
            param->data -= lr * param->grad;
        }
    }
}
