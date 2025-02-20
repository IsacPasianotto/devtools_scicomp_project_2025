from pyclassify.classifier import kNN
from pyclassify.utils import read_config, read_file
import argparse


TEST_SIZE=.2

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="experiments/config")
    return parser.parse_args()

def main(filename):
    kwargs = read_config(filename)
    dataset = kwargs['dataset']
    k = kwargs['k']
    
    x, y = read_file(dataset)
    idx_shuffle = [i for i in range(len(x))]
    x = [x[i] for i in idx_shuffle]
    y = [y[i] for i in idx_shuffle]
    test_size = int(len(x) * TEST_SIZE)

    x_train, x_test = x[:test_size], x[test_size:]
    y_train, y_test = y[:test_size], y[test_size:]
    
    knn = kNN(k)
    knn((x_train, y_train), x_test)

    pred_ok = sum([i == j for i, j in zip(y_test, knn.predicted)])
    print(f"Accuracy: {pred_ok / len(x_test)}")


if __name__ == "__main__":
    parser = parse_command_line_arguments()
    main(parser.config)
