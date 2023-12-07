from torch.utils.data import Dataset
import os


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, image_transform=None, label_transform=None):
        self.labels = READ_FILE(annotations_file)  # TODO
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, IDX_TO_IMAGE_NAME(idx))  # TODO
        image = READ_IMAGE(img_path)  # TODO, e.g., torchvision.io.read_image
        label = IDX_TO_GET_LABEL(self.labels, idx)  # TODO
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label


class TextDataset(Dataset):
    def __init__(self, data_file, text_transform=None, label_transform=None):
        self.texts, self.labels = READ_FILE_AND_SPLIT_DATA(data_file)  # TODO
        self.text_transform = text_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = IDX_TO_GET_TEXT(self.texts, idx)  # TODO
        label = IDX_TO_GET_LABEL(self.labels, idx)  # TODO
        if self.text_transform:
            text = self.text_transform(text)
        if self.label_transform:
            label = self.label_transform(label)
        return text, label

