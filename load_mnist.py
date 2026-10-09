import os
from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
os.makedirs("data", exist_ok=True)
def write(path, Xs, ys):
    with open(path, "w") as f:
        for xi, yi in zip(Xs, ys):
            f.write(str(int(yi)) + "," + ",".join(str(int(v)) for v in xi) + "\n")
write("data/mnist_train.csv", X[:60000], y[:60000])
write("data/mnist_test.csv", X[60000:], y[60000:])