#include <iostream>
#include <string>

#include <Eigen/Dense>
#include "cnpy.h"

#include "HopfieldNetwork.h"

using namespace std;

int main() {
//    string path = "data/mnist/original_28x28/binary_digits_binary_pixels/x_train.npy";
//    auto data = (read_npy_file(path) * 2).array() - 1;
//    HopfieldNetwork model(data);
//    model.save("binary_model.txt");
//    model.save_png("binary_model.png");

//    string path = "data/mnist/original_28x28/all_digits_binary_pixels/x_test.npy";
//    HopfieldNetwork model(read_npy_file(path));
//    model.save("full_model.txt");
//    model.save_png("full_model.png");

    HopfieldNetwork model("binary_model.txt");

    model.randomize_state();

    for (int i = 0; i < 150; i++) {
        model.update_state(25);
        model.save_state(format("inference/test{}.png", i));
    }

    return 0;
}
