#include "HopfieldNetwork.h"
#include "Util.h"

#include <fstream>
#include <iostream>
#include <random>

using namespace std;

HopfieldNetwork::HopfieldNetwork(const Eigen::MatrixXd &memory) {
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

    load_from_file(file, weights, n, n);

    file.close();
}

void HopfieldNetwork::randomize_state() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> random(-2, 2);

    state = Eigen::VectorXd(weights.rows());
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
    write_matrix_to_png(255 * normalize(weights), filename);
}

void HopfieldNetwork::save_state(const std::string &filename) const {
    int n = static_cast<int>(sqrt(state.rows()));
    assert(n * n == state.rows());

    auto image = state.reshaped(n, n).transpose();
    auto output = ((image.array() + 2) / 2.0);

    write_matrix_to_rgb(output, filename);
}
