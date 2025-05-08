#include "HopfieldNetwork.h"
#include "cnpy.h"
#include "stb_image_write.h"

#include <fstream>
#include <iostream>
#include <random>

using namespace std;

void write_matrix_to_png(const Eigen::MatrixXd &data, const string &path) {
    auto rows = data.rows();
    auto cols = data.cols();

    vector<uint8_t> output(data.size());
    for (auto i = 0; i < rows; i++)
        for (auto j = 0; j < cols; j++)
            output[j + cols * i] = static_cast<uint8_t>(data(i, j));

    stbi_write_png(path.c_str(),
                   static_cast<int>(rows), static_cast<int>(cols), 1, output.data(), static_cast<int>(cols));
}

void write_matrix_to_rgb(const Eigen::MatrixXd &data, const string &path) {
    auto rows = data.rows();
    auto cols = data.cols();

    vector<vector<int>> palette = {
            {240, 20},
            {150, 40},
            {50, 105}
    };

    vector<uint8_t> output(data.size() * 3);
    for (auto i = 0; i < rows; i++)
        for (auto j = 0; j < cols; j++)
            for (auto c = 0; c < 3; c++)
                output[3 * (j + cols * i) + c] =
                        palette[c][0] + (palette[c][1] - palette[c][0]) * static_cast<uint8_t>(data(i, j));

    stbi_write_png(path.c_str(),
                   static_cast<int>(rows), static_cast<int>(cols), 3, output.data(), 3 * static_cast<int>(cols));
}

Eigen::MatrixXd read_npy_file(const string &path) {
    try {
        cnpy::NpyArray npy_array = cnpy::npy_load(path);
        auto *data_ptr = npy_array.data<int16_t>();

        if (npy_array.shape.size() != 2)
            throw std::runtime_error("Unsupported dimensions");
        if (npy_array.word_size != 2)
            throw std::runtime_error("Unsupported data type");

        size_t rows = npy_array.shape[0];
        size_t cols = npy_array.shape[1];

        Eigen::MatrixXd data(rows, cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data(i, j) = data_ptr[i * cols + j];

        return data;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    std::cerr << "Unknown Error" << std::endl;
    exit(1);
}

HopfieldNetwork::HopfieldNetwork(const MatrixXd &memory) {
    cout << "Training Hopfield Network with " << memory.rows() << " rows\n";
    weights = (1.0 / static_cast<double>(memory.rows())) * memory.transpose() * memory;
    weights.diagonal().fill(0);
}

HopfieldNetwork::HopfieldNetwork(const std::string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    int n;
    file >> n;
    weights = MatrixXd(n, n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            file >> weights(i, j);

    file.close();
}

void HopfieldNetwork::randomize_state() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random(-2, 2);

    state = VectorXd(weights.rows());
    for (double &i: state)
        i = random(gen);
}

void HopfieldNetwork::update_state(int steps) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random(0, static_cast<int>(weights.rows()) - 1);

    for (int i = 0; i < steps; i++) {
        auto neuron = random(gen);
        auto activation = weights.col(neuron).dot(state);
        state[neuron] = activation >= 0 ? 1 : -1;
    }

    cout << -0.5 * state.transpose() * weights * state << '\n';
}

void HopfieldNetwork::save(const std::string &filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    file << weights.rows() << endl;
    file << weights << endl;
    file.close();
}

void HopfieldNetwork::save_png(const std::string &filename) const {
    auto min = weights.minCoeff();
    auto max = weights.maxCoeff();

    auto image = (weights.array() - min) / ((max - min) / 255.0);
    write_matrix_to_png(image, filename);
}

void HopfieldNetwork::save_state(const std::string &filename) const {
    int n = static_cast<int>(sqrt(state.rows()));
    assert(n * n == state.rows());

    auto min = state.minCoeff();
    auto max = state.maxCoeff();
    auto image = state.reshaped(n, n).transpose();
    auto output = ((image.array() + 2) / 2.0);

    write_matrix_to_rgb(output, filename);
}
