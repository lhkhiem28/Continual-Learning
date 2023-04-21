import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from models.models import PretextsCA
from engines import *

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/HAM/train/", 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 32, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../datasets/HAM/val/", 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 32, 
        shuffle = False, 
    ), 
}
model = PretextsCA(
    num_classes = 7, 
)
optimizer = torch.optim.Adam(
    model.parameters(), lr = 1e-4, 
)

save_ckp_dir = "../../ckps/HAM/PretextsCA"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
train_fn(
    train_loaders, num_epochs = 50, 
    model = model, 
    optimizer = optimizer, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    save_ckp_dir = save_ckp_dir, 
)