#include <iostream>
#include <string>

#include "HopfieldNetwork.h"

using namespace std;

int main() {
//    string path = "data/mnist/original_28x28/binary_digits_binary_pixels/x_train.npy";
//    auto data = (read_npy_file(path) * 2).array() - 1;
//    HopfieldNetwork model(data);
//    model.save("models/mnist.txt");
//    model.save_png("models/mnist.png");
//    HopfieldNetwork model("models/mnist.txt");  // load model

    string path = "data/food/food.npy";
    auto data = (read_npy_file(path) * 2).array() - 1;
    HopfieldNetwork model(data);
    model.save("models/food.txt");
    model.save_png("models/food.png");

    model.randomize_state();
    for (int i = 0; i < 150; i++) {
        model.update_state(30);
        model.save_state(format("inference/output{}.png", i));
    }

    return 0;
}
