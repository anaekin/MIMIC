from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob

from helpers.utils import ImageFolder


def compute_fid(real_dir, fake_dir, batch_size=24):
    fid = FrechetInceptionDistance(feature=768)
    real_dataset = ImageFolder(real_dir)
    fake_dataset = ImageFolder(fake_dir)

    real_loader = DataLoader(real_dataset, batch_size=batch_size)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size)
    for batch in real_loader:
        fid.update(batch, real=True)

    for batch in fake_loader:
        fid.update(batch, real=False)

    return {
        "FID": fid.compute().item(),
    }


def compute_inception_score(fake_dir, batch_size=32):
    inception = InceptionScore()
    fake_dataset = ImageFolder(fake_dir)

    fake_loader = DataLoader(fake_dataset, batch_size=batch_size)
    for batch in fake_loader:
        inception.update(batch)

    return {"IS": inception.compute().item()}
