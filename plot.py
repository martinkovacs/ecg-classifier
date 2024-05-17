import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(dataset_path: str = "dataset/mitbih_test.csv", index: int = 0) -> None:
    data = pd.read_csv(dataset_path, header=None, dtype=np.float32)
    results = data.iloc[:, -1]
    data.drop(data.columns[-1], axis=1, inplace=True)

    y = data.iloc[index]
    plt.plot(range(len(y)), y)

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4897569/table/tab1/?report=objectonly
    match results[index]:
        case 0:
            title = "Normal beat (N)"
        case 1:
            title = "Supraventricular ectopic beat (S)"
        case 2:
            title = "Ventricular ectopic beat (V)"
        case 3:
            title = "Fusion beat (F)"
        case 4:
            title = "Unknown beat (Q)"
        case _:
            title = ""

    plt.title(title)
    plt.show()
