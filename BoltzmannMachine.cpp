#include "BoltzmannMachine.h"
#include "Util.h"

#include <random>
#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

BoltzmannMachine::BoltzmannMachine(int visible, int hidden)
        : v(visible), a(visible),
          h(hidden), b(hidden),
          A(visible, visible), B(hidden, hidden), W(visible, hidden) {
    double stddev = 1.0 / sqrt(visible + hidden);
    gaussian_initialize(A, 0, stddev);
    gaussian_initialize(B, 0, stddev);
    gaussian_initialize(W, 0, stddev);
    gaussian_initialize(a, 0, stddev);
    gaussian_initialize(b, 0, stddev);

    A.diagonal().setZero();
    B.diagonal().setZero();
}

BoltzmannMachine::BoltzmannMachine(const std::string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    int visible, hidden;
    file >> visible >> hidden;

    load_from_file(file, A, visible, visible);
    load_from_file(file, B, hidden, hidden);
    load_from_file(file, W, visible, hidden);
    load_from_file(file, a, visible);
    load_from_file(file, b, hidden);

    v = VectorXd(visible);
    h = VectorXd(hidden);

    file.close();
}

void BoltzmannMachine::train(const MatrixXd &data, int epochs, int steps, double learning_rate) {
    MatrixXd best_A;
    MatrixXd best_B;
    MatrixXd best_W;
    VectorXd best_a;
    VectorXd best_b;
    double best_energy = numeric_limits<double>::max();

    for (int epoch = 0; epoch < epochs; epoch++) {
        // positive phase

        MatrixXd pos_A = MatrixXd::Zero(A.rows(), A.cols());
        MatrixXd pos_B = MatrixXd::Zero(B.rows(), B.cols());
        MatrixXd pos_W = MatrixXd::Zero(W.rows(), W.cols());
        VectorXd pos_a = VectorXd::Zero(a.size());
        VectorXd pos_b = VectorXd::Zero(b.size());

        for (int row = 0; row < data.rows(); row++) {
            v = data.row(row);

            pos_A += v * v.transpose();
            pos_a += v;
            pos_W += v * h.transpose();
            pos_b += h;
            pos_B += h * h.transpose();
        }

        pos_A /= data.rows();
        pos_a /= data.rows();
        pos_W /= data.rows();
        pos_b /= data.rows();
        pos_B /= data.rows();

        // negative phase

        MatrixXd neg_A = MatrixXd::Zero(A.rows(), A.cols());
        MatrixXd neg_B = MatrixXd::Zero(B.rows(), B.cols());
        MatrixXd neg_W = MatrixXd::Zero(W.rows(), W.cols());
        VectorXd neg_a = VectorXd::Zero(a.size());
        VectorXd neg_b = VectorXd::Zero(b.size());

        randomize_state();

        for (int step = 0; step < steps; step++) {
            update_state(1);
            neg_A += v * v.transpose();
            neg_a += v;
            neg_W += v * h.transpose();
            neg_b += h;
            neg_B += h * h.transpose();
        }

        neg_A /= steps;
        neg_a /= steps;
        neg_W /= steps;
        neg_b /= steps;
        neg_B /= steps;

        // update weights
        A += learning_rate * (pos_A - neg_A);
        B += learning_rate * (pos_B - neg_B);
        W += learning_rate * (pos_W - neg_W);
        a += learning_rate * (pos_a - neg_a);
        b += learning_rate * (pos_b - neg_b);

        A.diagonal().setZero();
        B.diagonal().setZero();

        double loss = energy();
        if (loss < best_energy) {
            best_energy = loss;
            best_A = A;
            best_B = B;
            best_W = W;
            best_a = a;
            best_b = b;
        }

        cout << "Epoch " << epoch + 1 << " / " << epochs << " | Energy: " << loss << "\n";
    }

    A = best_A;
    B = best_B;
    W = best_W;
    a = best_a;
    b = best_b;

    cout << "Training finished with final energy: " << best_energy << "\n";
}

void BoltzmannMachine::randomize_state() {
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution random(0.5);

    for (double &i: v)
        i = random(gen) ? 1 : -1;
    for (double &i: h)
        i = random(gen) ? 1 : -1;
}

void BoltzmannMachine::update_state(int steps) {
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<> random(0, static_cast<int>(v.size() + h.size()) - 1);

    for (int n = 0; n < steps; n++) {
        int i = random(rng);

        if (i < v.size()) {
            double p = probability_of_visible_on(i);
            v[i] = bernoulli_distribution(p)(rng) ? 1 : 0;
        }
        else {
            int j = i - static_cast<int>(v.size());
            double p = probability_of_hidden_on(j);
            h[j] = bernoulli_distribution(p)(rng) ? 1 : 0;
        }
    }
}

double BoltzmannMachine::energy() const {
    double e1 = 0.5 * v.dot(A * v);
    double e2 = 0.5 * h.dot(B * h);
    double e3 = v.dot(W * h);
    double e4 = a.dot(v) + b.dot(h);

    return -(e1 + e2 + e3 + e4);
}

double BoltzmannMachine::probability_of_visible_on(int i) const {
    if (h.size() == 0)
        return sigmoid(A.row(i).dot(v) + a[i]);
    return sigmoid(W.row(i).dot(h) + A.row(i).dot(v) + a[i]);
}

double BoltzmannMachine::probability_of_hidden_on(int j) const {
    return sigmoid(W.col(j).dot(v) + B.row(j).dot(h) + b[j]);
}

void BoltzmannMachine::save(const std::string &filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    file << A.rows() << ' ' << B.rows() << endl;
    file << A << B << W << a << b << endl;
    file.close();
}

void BoltzmannMachine::save_png(const std::string &filename) const {
    write_matrix_to_png(255 * normalize(A), filename + ".A.png");
    write_matrix_to_png(255 * normalize(B), filename + ".B.png");
    write_matrix_to_png(255 * normalize(W), filename + ".W.png");
}

void BoltzmannMachine::save_state(const std::string &filename) const {
    int n = static_cast<int>(sqrt(v.rows()));
    assert(n * n == v.rows());

    auto image = v.reshaped(n, n).transpose();

    write_matrix_to_rgb(image, filename);
}
