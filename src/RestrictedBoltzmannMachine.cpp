#include "RestrictedBoltzmannMachine.h"

#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

RestrictedBoltzmannMachine::RestrictedBoltzmannMachine(int visible, int hidden, RBMTrainParameters params)
        : n_visible(visible), n_hidden(hidden),
          x(n_visible), c(n_visible), h(n_hidden), b(n_hidden), W(n_visible, n_hidden),
          params(params) {
}

RestrictedBoltzmannMachine::RestrictedBoltzmannMachine(const std::string &filename)
        : n_hidden(0), n_visible(0) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    file >> n_hidden >> n_visible;

    load_from_file(file, W, n_visible, n_hidden);
    load_from_file(file, c, n_visible);
    load_from_file(file, b, n_hidden);

    file.close();

    x = VectorXd(n_visible);
    h = VectorXd(n_hidden);
    randomize_state();
}

void RestrictedBoltzmannMachine::train(const MatrixXd &data) {
    uniform_int_distribution<> index(0, data.rows() - 1);

    W_vel = MatrixXd::Zero(n_visible, n_hidden);
    b_vel = VectorXd::Zero(n_hidden);
    c_vel = VectorXd::Zero(n_visible);

    gaussian_initialize(W, 0, params.w_stddev);
    gaussian_initialize(c, params.xb_mean, params.xb_stddev);
    gaussian_initialize(b, params.hb_mean, params.hb_stddev);

    int batches = data.rows() / params.batch_size;
    for (int epoch = 0; epoch < params.epochs; epoch++) {
        double loss = 0;

        vector<int> indices(data.rows());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), rng);

        MatrixXd shuffled_data(data.rows(), data.cols());
        for (int i = 0; i < data.rows(); ++i)
            shuffled_data.row(i) = data.row(indices[i]);

        for (int i = 0; i < batches; i++) {
            auto batch = shuffled_data.block(i * params.batch_size, 0, params.batch_size, data.cols());

            loss += train_batch(batch);
            optimizer_step();
        }

        cout << "Epoch " << epoch + 1 << " / " << params.epochs << " | Loss: " << loss << endl;
    }
}

/**
 *
 * @param batch (batch_size, n_visible)
 */
double RestrictedBoltzmannMachine::train_batch(const Eigen::MatrixXd &batch) {
    // positive phase
    MatrixXd prob_of_h = probability_h_given_x(batch);  // (batch_size, n_hidden)
    MatrixXd h_hat = bernoulli_sample(prob_of_h);  // (batch_size, n_hidden)

    W_grad = batch.transpose() * prob_of_h;
    b_grad = prob_of_h.colwise().sum();  // (1, n_hidden)
    c_grad = batch.colwise().sum();  // (1, n_visible)

    // negative phase
    MatrixXd prob_of_x = probability_x_given_h(h_hat);  // (batch_size, n_visible)
    MatrixXd x_hat = bernoulli_sample(prob_of_x);  // (batch_size, n_visible)

    prob_of_h = probability_h_given_x(x_hat);  // (batch_size, n_hidden)

    W_grad -= x_hat.transpose() * prob_of_h;
    b_grad -= prob_of_h.colwise().sum();
    c_grad -= x_hat.colwise().sum();

    // normalizing
    int n = batch.rows();
    W_grad /= n;
    b_grad /= n;
    c_grad /= n;

    auto loss = (batch - x_hat).array().square().colwise().sum().mean();
    return loss;
}

void RestrictedBoltzmannMachine::optimizer_step() {
    W_vel *= params.momentum;
    b_vel *= params.momentum;
    c_vel *= params.momentum;

    W_vel += (1 - params.momentum) * params.learning_rate * W_grad;
    b_vel += (1 - params.momentum) * params.learning_rate * b_grad;
    c_vel += (1 - params.momentum) * params.learning_rate * c_grad;

    W += W_vel;
    b += b_vel;
    c += c_vel;
}

void RestrictedBoltzmannMachine::randomize_state() {
    for (double &i: x)
        i = uniform(rng) < 0.5 ? 1 : 0;
    for (double &i: h)
        i = uniform(rng) < 0.5 ? 1 : 0;
}

void RestrictedBoltzmannMachine::update_state(int steps) {
    for (int i = 0; i < steps; i++) {
        VectorXd probability_of_h = probability_h_given_x(x);
        h = bernoulli_sample(probability_of_h);

        VectorXd probability_of_x = probability_x_given_h(h);
        x = bernoulli_sample(probability_of_x);
    }
}

MatrixXd RestrictedBoltzmannMachine::probability_x_given_h(const MatrixXd &val) const {
    MatrixXd input = val * W.transpose();  // (batch_size, n_visible)
    input.rowwise() += c.transpose();
    return sigmoid(input);  // (batch_size, n_visible)
}

VectorXd RestrictedBoltzmannMachine::probability_x_given_h(const VectorXd &val) const {
    VectorXd input = W * val + c;  // (n_visible, )
    return sigmoid(input);
}

MatrixXd RestrictedBoltzmannMachine::probability_h_given_x(const MatrixXd &val) const {
    MatrixXd input = val * W;  // (batch_size, n_hidden)
    input.rowwise() += b.transpose();
    return sigmoid(input);   // (batch_size, n_hidden)
}

VectorXd RestrictedBoltzmannMachine::probability_h_given_x(const VectorXd &val) const {
    VectorXd input = W.transpose() * val + b;  // (n_hidden, )
    return sigmoid(input);
}

double RestrictedBoltzmannMachine::energy() const {
    return -h.dot(W * x) - c.dot(x) - b.dot(h);
}

double RestrictedBoltzmannMachine::free_energy() const {
    return -c.dot(x) - (W * x + b).array().exp().log1p().sum();
}

void RestrictedBoltzmannMachine::save(const string &filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    file << n_hidden << ' ' << n_visible << endl;
    file << W << endl;
    file << c << endl;
    file << b << endl;
    file.close();
}

void RestrictedBoltzmannMachine::save_weights_to_png(const string &filename) const {
    int n = static_cast<int>(sqrt(n_visible));
    int m = static_cast<int>(sqrt(n_hidden));

    MatrixXd weights = 255 * normalize(W);
    MatrixXd output(m * n, m * n);
    output.setZero();

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            if (j + i * m == n_hidden)
                goto end;
            MatrixXd block = weights.col(i * m + j).reshaped(n, n);
            output.block(i * n, j * n, n, n) = block.array();
        }

    end:
    write_matrix_to_png(output, filename);
}

void RestrictedBoltzmannMachine::save_state(const string &filename) const {
    int n = static_cast<int>(sqrt(n_visible));

    auto image = x.reshaped(n, n).transpose();
    write_matrix_to_rgb(image, filename);
}
