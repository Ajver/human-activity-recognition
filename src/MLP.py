import openml
from torch import nn

CLASS_LABELS = ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"]

dataset = openml.datasets.get_dataset(1478)
X, y, *_ = dataset.get_data(target=dataset.default_target_attribute)


class MLP(nn.Module):
    def __init__(self, n_hidden: int, n_width: int):
        super().__init__()

        self.input_layer = nn.Linear(X.shape[1], n_width)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_width, n_width) for _ in range(n_hidden)])
        self.output_layer = nn.Linear(n_width, len(CLASS_LABELS))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x