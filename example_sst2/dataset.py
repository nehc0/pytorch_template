from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data, text_transform=None, label_transform=None):
        self.texts = data['text']
        self.labels = data['label']
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
