#pragma once

#include <Eigen/Dense>

void write_matrix_to_png(const Eigen::MatrixXd &data, const std::string &path);

void write_matrix_to_rgb(const Eigen::MatrixXd &data, const std::string &path);

Eigen::MatrixXd read_npy_file(const std::string &path);
