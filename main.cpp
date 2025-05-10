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
            "5. Train Restricted Boltzmann Machine\n"
            "6. Load & Run Restricted Boltzmann Machine\n";
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
            string path = "data/mnist/mnist.npy";
            auto data = read_npy_file(path);

//            auto sample = data.row(10).reshaped(28, 28);
//            write_matrix_to_png(255 * sample, "sample.png");
//            exit(0);

            RBMTrainParameters params;
            params.learning_rate = 0.01;
            params.xb_mean = -0.2;
            params.hb_mean = -0.5;
            params.contrastive_divergence_steps = 1;
            params.epochs = 20;

            RestrictedBoltzmannMachine model(28 * 28, 100, params);

            model.train(data);
            model.save("models/mnist_rbm.txt");
            model.save_weights_to_png("models/mnist_rbm.png");
            break;
        }

        case 6: {
            RestrictedBoltzmannMachine model("models/mnist_rbm.txt");

            for (int i = 0; i < 1000; i++) {
                model.update_state(20);
                if (i % 100 == 0)
                    cout << i << " / 1000\n";
            }
            model.save_state("output.png");
            break;
        }

        default:
            exit(1);
    }

    return 0;
}
