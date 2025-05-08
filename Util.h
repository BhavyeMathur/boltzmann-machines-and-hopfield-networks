#pragma once

#include <fstream>
#include <Eigen/Dense>

void write_matrix_to_png(const Eigen::MatrixXd &data, const std::string &path);

void write_matrix_to_rgb(const Eigen::MatrixXd &data, const std::string &path);

Eigen::MatrixXd read_npy_file(const std::string &path);

void load_matrix_from_file(std::ifstream &file, Eigen::MatrixXd &matrix, int rows, int cols);

void load_vector_from_file(std::ifstream &file, Eigen::VectorXd &vector, int n);

Eigen::MatrixXd normalize(const Eigen::MatrixXd &matrix);
