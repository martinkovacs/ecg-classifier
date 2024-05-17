import numpy as np
import pandas as pd
import torch

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")


class ECGClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(180, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)
        )
    
    def forward(self, x):
        return self.layers(x)
    

def train(dataset_path: str = "dataset/mitbih_train.csv", model_path: str = "models/model.pt", epochs: int = 5000) -> None:
    data = pd.read_csv(dataset_path, header=None, dtype=np.float32)
    results = data.iloc[:, -1]
    data.drop(data.columns[-1], axis=1, inplace=True)

    X_train = torch.tensor(data.values, dtype=torch.float32)
    y_train = torch.nn.functional.one_hot(torch.tensor(results.values, dtype=torch.long)).to(torch.float32)
    model = ECGClassifier()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss}")

    torch.save(model.state_dict(), model_path)
