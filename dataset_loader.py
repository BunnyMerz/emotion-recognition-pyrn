from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import traceback
import importlib
import train


dataset = datasets.FER2013(
    root="data",
    split='train',
    transform=ToTensor(),
    # target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

data_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

def r(m):
    importlib.reload(m)

halt = 1
def exit():
    global halt
    halt = 0

while(halt):
    try:
        exec(input('>>> '))
        r(train)
    except Exception as e:
        print(traceback.format_exc())