#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <random>
#include "Util.h"

struct RBMTrainParameters {
    int epochs = 10;
    int batch_size = 256;
    int contrastive_divergence_steps = 10;

    double learning_rate = 0.001;
    double momentum = 0.01;

    double w_stddev = 0.1;
    double xb_mean = 0;
    double xb_stddev = 0.0;
    double hb_mean = 0;
    double hb_stddev = 0.0;
};

class RestrictedBoltzmannMachine {
public:
    RestrictedBoltzmannMachine(int visible, int hidden, RBMTrainParameters params);

    explicit RestrictedBoltzmannMachine(const std::string &filename);

    void train(const Eigen::MatrixXd &data);

    void randomize_state();

    void update_state(int steps);

    [[nodiscard]] double energy() const;

    [[nodiscard]] double free_energy() const;

    void save(const std::string &filename) const;

    void save_weights_to_png(const std::string &filename) const;

    void save_state(const std::string &filename) const;

private:
    int n_visible;  // number of visible units
    int n_hidden;  // number of hidden units

    Eigen::MatrixXd W;  // visible-to-hidden weights

    Eigen::VectorXd c;  // visible bias
    Eigen::VectorXd b;  // hidden bias

    Eigen::VectorXd x;  // visible state
    Eigen::VectorXd h;  // hidden state

    std::random_device rd;
    std::mt19937 rng{rd()};
    std::uniform_real_distribution<double> uniform{0, 1};

    // training variables
    Eigen::MatrixXd W_vel;
    Eigen::VectorXd c_vel;
    Eigen::VectorXd b_vel;

    Eigen::MatrixXd W_grad;
    Eigen::VectorXd c_grad;
    Eigen::VectorXd b_grad;

    RBMTrainParameters params;

    double train_batch(const Eigen::MatrixXd &batch);

    void optimizer_step();

    [[nodiscard]] Eigen::MatrixXd probability_x_given_h(const Eigen::MatrixXd &val) const;

    [[nodiscard]] Eigen::VectorXd probability_x_given_h(const Eigen::VectorXd &val) const;

    [[nodiscard]] Eigen::MatrixXd probability_h_given_x(const Eigen::MatrixXd &val) const;

    [[nodiscard]] Eigen::VectorXd probability_h_given_x(const Eigen::VectorXd &val) const;
};
