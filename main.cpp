#include <iostream>
#include <string>
#include <getopt.h>

#include "HopfieldNetwork.h"
#include "BoltzmannMachine.h"
#include "RestrictedBoltzmannMachine.h"
#include "Util.h"

using namespace std;

enum class Model {
    Hopfield,
    Boltzmann,
    RestrictedBoltzmann
};

struct Options {
    Model model = Model::RestrictedBoltzmann;
    int hidden = 100;

    bool train = true;
    string data_path;
    string name = "model";

    RBMTrainParameters params;
};

void printHelp() {
    std::cout << "Usage: ./main [OPTIONS]\n\n"
              << "Options:\n"
              << "  --help                  Show this help message and exit\n"
              << "  --model, -m <type>      Choose model type: 'hopfield', 'boltzmann', or 'rbm' (default: rbm)\n"
              << "  --train, -t <type>      'true' for training, 'false' for inference (default: true)\n"
              << "  --data, -d  <string>    Path to .npy file containing training data\n"
              << "  --name <string>         Output/input filename without extension (e.g. 'mnist_rbm')\n"
              << "  --epochs, -e <int>      Number of training epochs (e.g., 20)\n"
              << "  --batch_size, -b <int>  Batch size for training (e.g., 256)\n"
              << "  --hidden, -n <int>      Number of hidden neurons (e.g., 100)\n"
              << "  --cd_k, -k <int>        Number of Contrastive Divergence steps (e.g., 10)\n"
              << "  --lr, -l <float>        Learning rate (e.g., 0.05)\n"
              << "  --momentum, -p <float>  Momentum for gradient update (e.g., 0.5)\n"
              << "  --w_std, -w <float>     Standard deviation for initializing weights (e.g., 0.1)\n"
              << "  --x_mean, -x <float>    Mean for initializing visible bias (e.g., -0.2)\n"
              << "  --h_mean, -h <float>    Mean for initializing hidden bias (e.g., -0.5)\n"
              << "\nExamples:\n"
              << "  ./main --model rbm --output mnist_rbm --epochs 20 --batch_size 64 --cd_k 2 --lr 0.05 --momentum 0.5\n";
}

void getOptions(int argc, char **argv, Options &options) {
    opterr = static_cast<int>(false);

    option longOptions[] = {
            {"help",       no_argument,       nullptr, 'q'},
            {"model",      required_argument, nullptr, 'm'},
            {"train",      required_argument, nullptr, 't'},
            {"data",       required_argument, nullptr, 'd'},
            {"name",       required_argument, nullptr, 'o'},
            {"epochs",     required_argument, nullptr, 'e'},
            {"batch_size", required_argument, nullptr, 'b'},
            {"hidden",     required_argument, nullptr, 'n'},
            {"cd_k",       required_argument, nullptr, 'k'},
            {"lr",         required_argument, nullptr, 'l'},
            {"momentum",   required_argument, nullptr, 'p'},
            {"w_std",      required_argument, nullptr, 'w'},
            {"x_mean",     required_argument, nullptr, 'x'},
            {"h_mean",     required_argument, nullptr, 'h'},
            {nullptr, 0,                      nullptr, '\0'},
    };

    int choice;
    string arg;
    int index = 0;
    while ((choice = getopt_long(argc, argv, "qm:t:d:o:e:b:n:k:l:p:w:x:h:", static_cast<option *>(longOptions),
                                 &index)) != -1) {
        if (choice != 'q')
            arg = optarg;

        switch (choice) {
            case 'q':
                printHelp();
                exit(0);

            case 'm':
                if (arg == "hopfield")
                    options.model = Model::Hopfield;
                else if (arg == "boltzmann")
                    options.model = Model::Boltzmann;
                else if (arg == "rbm")
                    options.model = Model::RestrictedBoltzmann;
                else {
                    cerr << "Invalid model '" << arg << "'. Must be one of 'hopfield', 'boltzmann', or 'rbm'" << endl;
                    exit(1);
                }

                if (options.data_path.empty()) {
                    if (arg == "rbm") {
                        options.data_path = "data/mnist/mnist.npy";
                    }
                    else {
                        options.data_path = "data/food/food.npy";
                    }
                }
                break;

            case 't':
                options.train = arg == "true";
                break;

            case 'd':
                options.data_path = arg;
                break;

            case 'o':
                options.name = arg;
                break;

            case 'e':
                options.params.epochs = stoi(arg);
                break;

            case 'b':
                options.params.batch_size = stoi(arg);
                break;

            case 'n':
                options.hidden = stoi(arg);
                break;

            case 'k':
                options.params.contrastive_divergence_steps = stoi(arg);
                break;

            case 'l':
                options.params.learning_rate = stod(arg);
                break;

            case 'p':
                options.params.momentum = stod(arg);
                break;

            case 'w':
                options.params.w_stddev = stod(arg);
                break;

            case 'x':
                options.params.xb_mean = stod(arg);
                break;

            case 'h':
                options.params.hb_mean = stod(arg);
                break;

            default:
                cerr << "Unknown option" << endl;
                exit(1);
        }
    }
}

int main(int argc, char *argv[]) {
    cout << fixed << setprecision(4);

    Options options;
    getOptions(argc, argv, options);

    switch (options.model) {
        case Model::Hopfield: {
            if (options.train) {
                auto data = (read_npy_file(options.data_path) * 2).array() - 1;

                HopfieldNetwork model(data);
                model.save("models/" + options.name + ".txt");
                model.save_png("models/" + options.name + ".png");
            }
            else {
                HopfieldNetwork model("models/food.txt");

                for (int i = 0; i < 150; i++)
                    model.update_state(30);
                model.save_state(options.name + ".png");
            }

            break;
        }

        case Model::Boltzmann: {
            if (options.train) {
                auto data = (read_npy_file(options.data_path) * 2).array() - 1;

                BoltzmannMachine model(data.cols(), options.hidden);

                model.train(data, options.params.epochs,
                            options.params.contrastive_divergence_steps,
                            options.params.learning_rate);

                model.save("models/" + options.name + ".txt");
                model.save_png("models/" + options.name);
            }
            else {
                BoltzmannMachine model("models/" + options.name + ".txt");

                for (int i = 0; i < 150; i++)
                    model.update_state(1);
                model.save_state(options.name + ".png");
            }

            break;
        }

        case Model::RestrictedBoltzmann: {
            if (options.train) {
                Eigen::MatrixXd data = read_npy_file(options.data_path) / 255;

                RestrictedBoltzmannMachine model(data.cols(), options.hidden, options.params);

                model.train(data);
                model.save("models/" + options.name + ".txt");
                model.save_weights_to_png("models/" + options.name + ".png");
            }
            else {
                RestrictedBoltzmannMachine model("models/" + options.name + ".txt");

                for (int i = 0; i < 100; i++)
                    model.update_state(1);
                model.save_state(options.name + ".png");
            }

            break;
        }
    }

    return 0;
}
