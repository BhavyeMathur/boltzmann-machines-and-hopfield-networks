#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>

class RestrictedBoltzmannMachine {
public:
    RestrictedBoltzmannMachine(int visible, int hidden);

    explicit RestrictedBoltzmannMachine(const std::string &filename);

    void train(const Eigen::MatrixXd &data, int epochs, int batch_size = 64, double learning_rate = 0.005);

    void randomize_state();

    void update_state(int steps);

    [[nodiscard]] double energy() const;

    void save(const std::string &filename) const;

    void save_weights_to_png(const std::string &filename) const;

    void save_state(const std::string &filename) const;

private:
    int V;  // number of visible units
    int H;  // number of hidden units

    Eigen::MatrixXd W;  // visible-to-hidden weights

    Eigen::VectorXd c;  // visible bias
    Eigen::VectorXd b;  // hidden bias

    Eigen::VectorXd x;  // visible state
    Eigen::VectorXd h;  // hidden state

    [[nodiscard]] double probability_of_visible_on(int i) const;

    [[nodiscard]] double probability_of_hidden_on(int j) const;

    void gibbs_sample(const Eigen::VectorXd &x0, int steps);

    void gibbs_sample_hidden();

    void gibbs_sample_visible();
};
