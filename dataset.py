from torch.utils.data import Dataset
import os


class ImageDataset(Dataset):
    def __init__(self, data, image_transform=None, label_transform=None):
        self.images = data['image']  # TODO
        self.labels = data['label']  # TODO
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label


class TextDataset(Dataset):
    def __init__(self, data, text_transform=None, label_transform=None):
        self.texts = data['text']  # TODO
        self.labels = data['label']  # TODO
        self.text_transform = text_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.text_transform:
            text = self.text_transform(text)
        if self.label_transform:
            label = self.label_transform(label)
        return text, label
