#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>

class BoltzmannMachine {
public:
    BoltzmannMachine(int visible, int hidden);

    BoltzmannMachine(const std::string &filename);

    void train(const Eigen::MatrixXd &data, int epochs, int steps, double learning_rate);

    void randomize_state();

    void update_state(int steps);

    [[nodiscard]] double energy() const;

    void save(const std::string &filename) const;

    void save_state(const std::string &filename) const;

private:
    Eigen::MatrixXd A;  // visible-to-visible weights
    Eigen::MatrixXd B;  // hidden-to-hidden weights
    Eigen::MatrixXd W;  // visible-to-hidden weights

    Eigen::VectorXd a;  // visible bias
    Eigen::VectorXd b;  // hidden bias

    Eigen::VectorXd v;  // visible state
    Eigen::VectorXd h;  // hidden state

    [[nodiscard]] double probability_of_visible_on(int i) const;

    [[nodiscard]] double probability_of_hidden_on(int j) const;
};
