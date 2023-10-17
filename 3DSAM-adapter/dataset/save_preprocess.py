import pickle
import os


def spacing(path_prefix=None):
    data_dir = os.path.join(path_prefix, "split.pkl")
    with open(data_dir, "rb") as f:
        d = pickle.load(f)[0]["train"]
    print(path_prefix, len(d.keys()))
    img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]


if __name__ == "__main__":
    spacing(path_prefix="D:\\ds\\kist_update\\data")