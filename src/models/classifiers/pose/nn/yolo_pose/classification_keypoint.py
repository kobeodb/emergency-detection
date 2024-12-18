import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(
            self,
            input_size=24,
            hidden_size=256,
            num_classes=2
    ):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out



class KeypointClassification:
    def __init__(self, path_model):
        self.path_model = path_model
        self.classes = ['0', '1']
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.model = NeuralNet()
        self.model.load_state_dict(
            torch.load(self.path_model, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, input_keypoint):
        if not isinstance(input_keypoint, torch.Tensor):
            input_keypoint = torch.tensor(
                input_keypoint, dtype=torch.float32
            )
        input_keypoint = input_keypoint.to(self.device)

        # with torch.no_grad():
            # out = self.model(input_keypoint)
            # _, predict = torch.max(out, -1)
            # label_predict = self.classes[predict]
            # return label_predict

        with torch.no_grad():
            out = self.model(input_keypoint)
            probabilities = F.softmax(out, dim=0)

            # _, predict = torch.max(probabilities, -1)
            # label_predict = self.classes[predict]
            return probabilities.cpu().numpy()