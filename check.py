import PIL
import torch, torchvision
from torchvision.transforms import ToTensor, Lambda
from train import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def main():
    model = torch.load('models/hm_9000_detect.30e.pth')
    model.eval()  # Set to eval mode to change behavior of Dropout, BatchNorm

    classes =  ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print(model,'\n')

    # Abrir uma imagem e transformá-la conforme a especificação de resolução de entrada e
    # normalização de valores definios na especifiação rede escolhida e em seu treinamento.
    # Essa imagem inclui um cão da raça Samoyed.
    input_image = PIL.Image.open('tests/fear1.png').convert('L')
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.CenterCrop(48),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485], std=[0.225]),
    ])
    input_tensor = preprocess(input_image)
    imshow(input_tensor)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    for c, p in zip(classes, probabilities):
        print(f'{100*p:1.2f}%: {c}')
    print()


if __name__ == '__main__':
    main()
    input()
