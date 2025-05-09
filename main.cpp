#include <iostream>
#include <string>

#include "HopfieldNetwork.h"
#include "BoltzmannMachine.h"
#include "RestrictedBoltzmannMachine.h"
#include "Util.h"

using namespace std;

int main() {
    int option;
    cout << "1. Train new Hopfield Network\n"
            "2. Load & Run Hopfield Network\n"
            "3. Train new Boltzmann Machine\n"
            "4. Load & Run Boltzmann Machine\n"
            "5. Train Restricted Boltzmann Machine\n";
    cin >> option;

    switch (option) {
        case 1: {
            string path = "data/food/food.npy";
            auto data = (read_npy_file(path) * 2).array() - 1;

            HopfieldNetwork model(data);
            model.save("models/food.txt");
            break;
        }

        case 2: {
            HopfieldNetwork model("models/food.txt");
            model.save_png("models/food_hopfield.png");

            model.randomize_state();
            for (int i = 0; i < 150; i++) {
                model.update_state(30);
                model.save_state(format("inference/output{}.png", i));
            }
            break;
        }

        case 3: {
            BoltzmannMachine model(24 * 24, 16 * 16);

            string path = "data/food/food.npy";
            auto data = (read_npy_file(path) * 2).array() - 1;
            model.train(data, 200, 50, 0.01);
            model.save("models/food_bm.txt");
            model.save_png("models/food_bm");

            break;
        }

        case 4: {
            BoltzmannMachine model("models/food_bm.txt");
            model.set_temperature(5);

            model.randomize_state();
            for (int i = 0; i < 150; i++) {
                model.update_state(20);
                model.save_state(format("inference/output{}.png", i));
            }

            cout << "Energy: " << model.energy() << '\n';
            break;
        }

        case 5: {
            RestrictedBoltzmannMachine model(28 * 28, 1000);

            string path = "data/mnist/original_28x28/all_digits_binary_pixels/x_train.npy";
            auto data = (read_npy_file(path) * 2).array() - 1;

            model.train(data, 100, 128, 0.001);
            model.save("models/mnist_rbm.txt");
            model.save_weights_to_png("models/mnist_rbm.png");
            break;
        }

        case 6: {
            RestrictedBoltzmannMachine model("models/mnist_rbm.txt");

            model.randomize_state();
            for (int i = 0; i < 150; i++) {
                model.update_state(20);
                model.save_state(format("inference/output{}.png", i));
            }
            break;
        }

        default:
            exit(1);
    }

    return 0;
}
