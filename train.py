import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import dataset_load

device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.features = nn.Sequential(
            ## [b, 1, 48, 48]
            nn.Conv2d(1, 32, kernel_size=3, stride=1), ## [batch, 32, 46, 46]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), ## [batch, 64, 44, 44]
            nn.ReLU(),
            nn.MaxPool2d(2), ## [batch, 64, 22, 22]
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1), ## [batch, 128, 20, 20]
            nn.ReLU(),
            nn.MaxPool2d(2), ## [batch, 128, 10, 10]
            nn.Conv2d(128, 128, kernel_size=3, stride=1), ## [batch, 128, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2), ## [batch, 128, 4, 4]
            nn.Dropout(0.25),
        )
        self.flatten = nn.Flatten() ## [batch, 128*4*4]
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,7),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

#########

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, epoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main(train_dataloader,validation_dataloader,test_dataloader,model,epochs=5):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=emotion_model.parameters(), lr=0.001)
    
    try:
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, t)
            test_loop(validation_dataloader, model, loss_fn, t)
        print("Done!")
    finally:
        test_loop(test_dataloader, model, loss_fn, -1)
        torch.save(emotion_model, 'pyrn-e45.pth')


if __name__=='__main__':
    # emotion_model = NeuralNetwork().to(device)
    emotion_model = torch.load('models/pyrn-e40.pth')

    train_dataset = dataset_load.FER2013Custom(
        root="data",
        split="train",
        transform=ToTensor(),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )


    validation_dataset = dataset_load.FER2013Custom(
        root="data",
        split="validation",
        transform=ToTensor(),
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=128,
        shuffle=True
    )


    test_dataset = dataset_load.FER2013Custom(
        root="data",
        split="test",
        transform=ToTensor(),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True
    )

    main(train_dataloader, validation_dataloader, test_dataloader, emotion_model)
