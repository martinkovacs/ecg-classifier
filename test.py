import numpy as np
import pandas as pd
import torch

from train import ECGClassifier

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")


def test(dataset_path: str = "dataset/mitbih_test.csv", model_path: str = "models/model.pt") -> None:
    model = ECGClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_data = pd.read_csv(dataset_path, header=None, dtype=np.float32)
    test_results = test_data.iloc[:, -1]
    test_data.drop(test_data.columns[-1], axis=1, inplace=True)

    with torch.inference_mode():    
        pred = torch.nn.functional.softmax(model(torch.tensor(test_data.values, dtype=torch.float32)), dim=1)

        correct = [0, 0, 0, 0, 0]
        total = [0, 0, 0, 0, 0]
        for idx, p in enumerate(pred):
            if int(torch.max(p, 0).indices.item()) == int(test_results[idx]):
                correct[int(test_results[idx])] += 1
            total[int(test_results[idx])] += 1
            
        print(sum(correct), len(pred), sum(correct) / len(pred))
        print([c / t for c, t in zip(correct, total)])
        print([(c, t) for c, t in zip(correct, total)])
