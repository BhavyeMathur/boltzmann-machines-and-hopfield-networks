#include "Util.h"

#include "stb_image_write.h"
#include "cnpy.h"

using namespace std;

void write_matrix_to_png(const Eigen::MatrixXd &data, const string &path) {
    auto rows = data.rows();
    auto cols = data.cols();

    vector<uint8_t> output(data.size());
    for (auto i = 0; i < rows; i++)
        for (auto j = 0; j < cols; j++)
            output[j + cols * i] = static_cast<uint8_t>(data(i, j));

    stbi_write_png(path.c_str(),
                   static_cast<int>(rows), static_cast<int>(cols), 1, output.data(), static_cast<int>(cols));
}

void write_matrix_to_rgb(const Eigen::MatrixXd &data, const string &path) {
    auto rows = data.rows();
    auto cols = data.cols();

    vector<vector<int>> palette = {
            {240, 20},
            {150, 40},
            {50,  105}
    };

    vector<uint8_t> output(data.size() * 3);
    for (auto i = 0; i < rows; i++)
        for (auto j = 0; j < cols; j++)
            for (auto c = 0; c < 3; c++)
                output[3 * (j + cols * i) + c] =
                        palette[c][0] + (palette[c][1] - palette[c][0]) * static_cast<uint8_t>(data(i, j));

    stbi_write_png(path.c_str(),
                   static_cast<int>(rows), static_cast<int>(cols), 3, output.data(), 3 * static_cast<int>(cols));
}

Eigen::MatrixXd read_npy_file(const string &path) {
    try {
        cnpy::NpyArray npy_array = cnpy::npy_load(path);
        auto *data_ptr = npy_array.data<int16_t>();

        if (npy_array.shape.size() != 2)
            throw std::runtime_error("Unsupported dimensions");
        if (npy_array.word_size != 2)
            throw std::runtime_error("Unsupported data type");

        size_t rows = npy_array.shape[0];
        size_t cols = npy_array.shape[1];

        Eigen::MatrixXd data(rows, cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data(i, j) = data_ptr[i * cols + j];

        return data;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    std::cerr << "Unknown Error" << std::endl;
    exit(1);
}

void load_matrix_from_file(ifstream &file, Eigen::MatrixXd &matrix, int rows, int cols) {
    matrix = Eigen::MatrixXd(rows, cols);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            file >> matrix(i, j);
}

void load_vector_from_file(std::ifstream &file, Eigen::VectorXd &vector, int n) {
    vector = Eigen::VectorXd(n);

    for (int i = 0; i < n; i++)
        file >> vector[i];
}
