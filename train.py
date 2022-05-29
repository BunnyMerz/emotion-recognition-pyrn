import torch
from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# import torchvision.models as models
# from torchvision.transforms import ToTensor, Lambda
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"

def epoch_info(epoch, batch, cost, avg_cost, acc):
    print(f"Epoch= {epoch}, batch = {batch}, cost = {cost}, accuracy = {acc}")
    print(f"[Epoch: {epoch}], averaged cost = {avg_cost}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
    
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main(data_loader):
    emotion_model = NeuralNetwork().to(device)
    batch_size = 16
    total_batch = len(data_loader) // batch_size
    epoch_size = 10

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=emotion_model.parameters(), lr=0.01)

    train_cost = []
    train_accu = []
    avg_cost = 0
        
    for epoch in range(epoch_size):
        for i, (image_batch, emotion_batch) in enumerate(data_loader):
            images = Variable(image_batch)
            emotions = Variable(emotion_batch)

            optimizer.zero_grad()
            hypothesis = emotion_model(images)
            cost = loss(hypothesis, emotions)
            cost.backward()
            optimizer.step()

            prediction = hypothesis.data.max(dim=1)[1]
            train_accu.append(((prediction.data == emotions.data).float().mean()).item())
            train_cost.append(cost.item())

            if i % 10 == 0:
                avg_cost = cost.data / total_batch
                epoch_info(epoch, i, train_cost[-1], avg_cost, train_accu[-1])


    print(train_cost)
    print(train_accu)

# model = torch.load('model.pth')
# torch.save(model, 'model.pth')
# model.eval()
