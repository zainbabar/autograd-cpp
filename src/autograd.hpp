#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "value.hpp"

// Each node's backward only pushes grad one level down to its inputs, so to backprop the whole
// graph with a single call we need to run them in the right order.
// Topological sort: DFS that adds a node to the list only after all of its inputs are added,
// so every node comes strictly after its inputs.

// recursive helper: visited and order are shared across every call
inline void visit(const std::shared_ptr<Value>& node,
           std::unordered_set<Value*>& visited,
           std::vector<std::shared_ptr<Value>>& order) {
    // a node can be reached through more than one path, only add it once
    if (visited.contains(node.get())) { return; }
    visited.insert(node.get());
    for (const auto& input : node->inputs) {
        visit(input, visited, order);
    }
    order.push_back(node);
}

// entry point: creates the shared state once and returns the order
inline std::vector<std::shared_ptr<Value>> build(const std::shared_ptr<Value>& output) {
    std::unordered_set<Value*> visited;
    std::vector<std::shared_ptr<Value>> order;
    visit(output, visited, order);
    return order;
}

// Grads aren't reset first, so calling this twice on the same graph accumulates.
inline void backprop(const std::shared_ptr<Value>& output) {
    std::vector<std::shared_ptr<Value>> order = build(output);
    output->grad = 1; // d(output)/d(output), seeds the chain rule
    // Walk the order in reverse (output first). By the time a node's backward runs, every node
    // that used it as an input has already pushed its grad in, so its grad is complete.
    for (auto nodeIt = order.rbegin(); nodeIt != order.rend(); ++nodeIt) {
        std::shared_ptr<Value>& node = *nodeIt;
        node->backward();
    }
}
