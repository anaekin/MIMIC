import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import random

from torchvision import transforms
from torchvision.transforms import PILToTensor
from torch.utils.data import DataLoader


class ImageNetDataloader(DataLoader):
    """
    Creates a dataloader for the target class from the Imagenet dataset

    Args:
        dataset: An instance of `ImageNetDataset`.
        batch_size (optional): The batch size (default: `32`).
        **kwargs: Additional Dataloader keyword arguments to pass to the dataloader.
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        **kwargs,
    ):
        self.batch_size = batch_size

        super().__init__(
            dataset,
            batch_size=self.batch_size,
            **kwargs,
        )


class ImageNetDataset(Dataset):
    """
    PyTorch Dataset for loading images from multiple ImageNet classes.

    Args:
        class_indices: List of indices from LOC_synset_mapping.txt.
        root_dir: Root directory of the ImageNet dataset.
        split (optional): 'train' or 'val' split.
        transform (optional): Transformations to apply to the images.
    """

    def __init__(
        self, class_index, root_dir, split="train", transform=None, max_per_class=None
    ):
        self.class_indices = (
            class_index if isinstance(class_index, list) else [class_index]
        )
        self.root_dir = root_dir
        self.split = split
        self.max_per_class = max_per_class
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((336, 336)),
                transforms.PILToTensor(),
            ]
        )

        self.synset_id_map = self._load_synset_ids()
        self.image_paths, self.labels = self._collect_all_image_paths()

    def _load_synset_ids(self):
        mapping_file = os.path.join(self.root_dir, "LOC_synset_mapping.txt")
        with open(mapping_file, "r") as f:
            lines = f.readlines()
        return {idx: lines[idx].strip().split(" ", 1)[0] for idx in self.class_indices}

    def _collect_all_image_paths(self):
        all_paths = []
        all_labels = []

        for class_index, synset_id in self.synset_id_map.items():
            class_folder = os.path.join(
                self.root_dir, "ILSVRC", "Data", "CLS-LOC", self.split, synset_id
            )
            filenames = [
                fname
                for fname in os.listdir(class_folder)
                if fname.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if self.max_per_class:
                filenames = filenames[: self.max_per_class]

            for fname in filenames:
                all_paths.append(os.path.join(class_folder, fname))
                all_labels.append(class_index)

        return all_paths, all_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    def save_single_image(self, class_idx, img_idx=0, save_path="sample.png"):
        class_image_paths = [
            path
            for path, label in zip(self.image_paths, self.labels)
            if label == class_idx
        ]

        if img_idx >= len(class_image_paths):
            raise IndexError(
                f"Class {class_idx} has only {len(class_image_paths)} images."
            )

        image = Image.open(class_image_paths[img_idx]).convert("RGB")
        image.save(save_path)
