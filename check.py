import PIL
import torch, torchvision
from torchvision.transforms import ToTensor, Lambda
from train import NeuralNetwork

def main():
    model = torch.load('emotion_model.pth')
    model.eval()  # Set to eval mode to change behavior of Dropout, BatchNorm

    classes =  ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print(model,'\n')

    # Abrir uma imagem e transformá-la conforme a especificação de resolução de entrada e
    # normalização de valores definios na especifiação rede escolhida e em seu treinamento.
    # Essa imagem inclui um cão da raça Samoyed.
    input_image = PIL.Image.open('happy.png').convert('L')
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.CenterCrop(48),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485], std=[0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    for c, p in zip(classes, probabilities):
        print(f'{100*p:1.2f}%: {c}')
    print()


if __name__ == '__main__':
    main()
