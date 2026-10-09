#pragma once

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// one row per image. pixels are scaled from 0-255 down to [0, 1]
struct Dataset {
    std::vector<std::vector<double>> pixels;
    std::vector<int> labels;
};

// reads a CSV where each row is: label, then the pixel values.
// max_rows = 0 means read everything, otherwise only the first max_rows rows
inline Dataset load_csv(const std::string& path, size_t max_rows = 0) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "can't open " << path << std::endl;
        std::exit(1);
    }
    Dataset data;
    std::string line;
    size_t n_cols = 0;
    while (std::getline(file, line)) {
        if (line.empty()) { continue; }
        if (!std::isdigit(static_cast<unsigned char>(line[0]))) { continue; }  // skip a header row
        std::stringstream row(line);
        std::string field;
        std::getline(row, field, ',');
        data.labels.push_back(std::stoi(field));
        std::vector<double> px;
        while (std::getline(row, field, ',')) {
            px.push_back(std::stod(field) / 255.0);
        }
        if (n_cols == 0) { n_cols = px.size(); }
        if (px.size() != n_cols) {
            std::cerr << "row " << data.labels.size() << " has " << px.size()
                      << " pixels, expected " << n_cols << std::endl;
            std::exit(1);
        }
        data.pixels.push_back(px);
        if (max_rows != 0 && data.labels.size() >= max_rows) { break; }
    }
    return data;
}

// draws one image in the terminal so you can check it matches its label
inline void print_digit(const std::vector<double>& px, int side = 28) {
    const std::string shades = " .:-=+*#%@";
    for (int r = 0; r < side; ++r) {
        for (int c = 0; c < side; ++c) {
            int level = static_cast<int>(px[r * side + c] * 9.999);
            std::cout << shades[level] << shades[level];  // doubled so it looks square
        }
        std::cout << "\n";
    }
}