#pragma once

#include <fstream>
#include <Eigen/Dense>

void write_matrix_to_png(const Eigen::MatrixXd &data, const std::string &path);

void write_matrix_to_rgb(const Eigen::MatrixXd &data, const std::string &path);

Eigen::MatrixXd read_npy_file(const std::string &path);

void load_from_file(std::ifstream &file, Eigen::MatrixXd &matrix, int rows, int cols);

void load_from_file(std::ifstream &file, Eigen::VectorXd &vector, int n);

void gaussian_initialize(Eigen::MatrixXd &matrix, double mean, double stddev);

void gaussian_initialize(Eigen::VectorXd &matrix, double mean, double stddev);

Eigen::MatrixXd normalize(const Eigen::MatrixXd &matrix);

double sigmoid(double x);

Eigen::VectorXd sigmoid(const Eigen::VectorXd &x);
