import sys
import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    mpl.style.use("fivethirtyeight")
except:
    pass


def plot_net(n_per_layer):
    no_layers = len(n_per_layer)
    f, ax = plt.subplots()

    x = 0
    for l in range(no_layers):
        neurons = int(n_per_layer[l])
        y = (int(max(n_per_layer)) - neurons) / 2.0
        for n in range(neurons):
            ax.scatter(x, y, color="k", s=200, alpha=0.6)
            y += 1
        x += 2

    plt.show()


def main(input_path):
    with open(input_path, "r") as f:
        n_per_layer = [l.strip()for l in f.readlines()]
        plot_net(n_per_layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to net dump")

    args = parser.parse_args()
    if getattr(args, "input") is None:
        parser.print_usage()
        sys.exit()

    main(getattr(args, "input"))
