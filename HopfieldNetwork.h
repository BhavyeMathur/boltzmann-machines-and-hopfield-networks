#pragma once

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class HopfieldNetwork {
public:
    explicit HopfieldNetwork(const MatrixXd &memory);

    explicit HopfieldNetwork(const string &filename);

    void randomize_state();

    void update_state(int steps);

    void save(const string &filename) const;

    void save_png(const std::string &filename) const;

    void save_state(const std::string &filename) const;

private:
    MatrixXd weights;
    VectorXd state;
};

void write_matrix_to_png(const Eigen::MatrixXd &data, const string &path);

Eigen::MatrixXd read_npy_file(const string &path);
