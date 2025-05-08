#pragma once

#include <Eigen/Dense>

class HopfieldNetwork {
public:
    explicit HopfieldNetwork(const Eigen::MatrixXd &memory);

    explicit HopfieldNetwork(const std::string &filename);

    void randomize_state();

    void update_state(int steps);

    void save(const std::string &filename) const;

    void save_png(const std::string &filename) const;

    void save_state(const std::string &filename) const;

private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd state;
};
