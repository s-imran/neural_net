import os
import sys
import argparse
from sklearn import datasets


def main(output_path, plot):
    X, y = datasets.make_moons(200, noise=0.20)

    if plot is not False:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        try:
            mpl.style.use("fivethirtyeight")
        except:
            pass
        f, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], s=100, c=y, alpha=0.6)
        plt.show()

    with open(output_path, "w") as f:
        for i in range(len(X)):
            f.write("{},{},{}\n".format(X[i, 0], X[i, 1], y[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Path to output data")
    parser.add_argument("--plot", action="store_true",
                        default=False, help="Plot dataset")

    args = parser.parse_args()
    if getattr(args, "output") == None:
        parser.print_usage()
        sys.exit()

    main(getattr(args, "output"), getattr(args, "plot"))
