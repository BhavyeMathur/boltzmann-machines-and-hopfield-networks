#include "RestrictedBoltzmannMachine.h"
#include "Util.h"

#include <random>
#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

RestrictedBoltzmannMachine::RestrictedBoltzmannMachine(int visible, int hidden)
        : V(visible), H(hidden),
          x(visible), c(visible),
          h(hidden), b(hidden),
          W(hidden, visible) {
    double stddev = 1.0 / sqrt(visible + hidden);
    gaussian_initialize(W, 0, stddev);
}

RestrictedBoltzmannMachine::RestrictedBoltzmannMachine(const std::string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    int visible, hidden;
    file >> visible >> hidden;

    load_from_file(file, W, visible, hidden);
    load_from_file(file, c, visible);
    load_from_file(file, b, hidden);

    file.close();

    x = VectorXd(visible);
    h = VectorXd(hidden);
}

void RestrictedBoltzmannMachine::train(const MatrixXd &data, int epochs, int batch_size, double learning_rate) {
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<> index(0, data.rows() - 1);

    for (int epoch = 0; epoch < epochs; epoch++) {
        // positive phase
        MatrixXd pos_W = MatrixXd::Zero(W.rows(), W.cols());
        VectorXd pos_c = VectorXd::Zero(c.size());
        VectorXd pos_b = VectorXd::Zero(b.size());

        for (int i = 0; i < batch_size; i++) {
            int row = index(rng);

            x = data.row(row);
            gibbs_sample_hidden();

            pos_W += h * x.transpose();
            pos_b += h;
            pos_c += x;
        }

        pos_W /= data.rows();
        pos_b /= data.rows();
        pos_c /= data.rows();

        // negative phase
        MatrixXd neg_W = MatrixXd::Zero(W.rows(), W.cols());
        VectorXd neg_c = VectorXd::Zero(c.size());
        VectorXd neg_b = VectorXd::Zero(b.size());

        for (int i = 0; i < batch_size; i++) {
            int row = index(rng);

            x = data.row(row);
            gibbs_sample(x, 10);
            auto hidden_probabilities = sigmoid(b + W * x);

            neg_W += hidden_probabilities * x.transpose();
            neg_b += hidden_probabilities;
            neg_c += x;
        }

        neg_W /= data.rows();
        neg_b /= data.rows();
        neg_c /= data.rows();

        // parameter updating

        W += learning_rate * (pos_W - neg_W);
        b += learning_rate * (pos_b - neg_b);
        c += learning_rate * (pos_c - neg_c);

        cout << "Epoch " << epoch + 1 << " / " << epochs << " | Energy: " << energy() << "\n";
    }
}

void RestrictedBoltzmannMachine::randomize_state() {
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution random(0.5);

    for (double &i: x)
        i = random(gen) ? 1 : 0;
    for (double &i: h)
        i = random(gen) ? 1 : 0;
}

void RestrictedBoltzmannMachine::update_state(int steps) {
    gibbs_sample(x, steps);
}

void RestrictedBoltzmannMachine::gibbs_sample(const VectorXd &x0, int steps) {
    x = x0;

    for (int i = 0; i < steps; i++) {
        gibbs_sample_hidden();
        gibbs_sample_visible();
    }
}

void RestrictedBoltzmannMachine::gibbs_sample_hidden() {
    random_device rd;
    mt19937 rng(rd());

    for (int j = 0; j < h.size(); j++) {
        double p = probability_of_hidden_on(j);
        h[j] = bernoulli_distribution(p)(rng) ? 1 : 0;
    }
}

void RestrictedBoltzmannMachine::gibbs_sample_visible() {
    random_device rd;
    mt19937 rng(rd());

    for (int k = 0; k < x.size(); k++) {
        double p = probability_of_visible_on(k);
        x[k] = bernoulli_distribution(p)(rng) ? 1 : 0;
    }
}

double RestrictedBoltzmannMachine::energy() const {
    double e1 = h.dot(W * x);
    double e2 = c.dot(x) + b.dot(h);
    return -(e1 + e2);
}

double RestrictedBoltzmannMachine::probability_of_visible_on(int k) const {
    double input = W.col(k).dot(h) + c[k];
    return sigmoid(input);
}

double RestrictedBoltzmannMachine::probability_of_hidden_on(int j) const {
    double input = W.row(j).dot(x) + b[j];
    return sigmoid(input);
}

void RestrictedBoltzmannMachine::save(const string &filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    file << W.rows() << ' ' << W.cols() << endl;
    file << W << c << b << endl;
    file.close();
}

void RestrictedBoltzmannMachine::save_weights_to_png(const string &filename) const {
    write_matrix_to_png(255 * normalize(W), filename);
}

void RestrictedBoltzmannMachine::save_state(const string &filename) const {
    int n = static_cast<int>(sqrt(x.rows()));

    auto image = x.reshaped(n, n).transpose();
    write_matrix_to_rgb(image, filename);
}
