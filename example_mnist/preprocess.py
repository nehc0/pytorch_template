import torch
from torchvision.transforms import v2


def preprocess_cv():
    """a function for preprocessing in CV task

    return image_transform, label_transform
    
    """

    # image transform
    image_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),

        v2.RandomRotation(degrees=(-60, 60)),
        v2.RandomHorizontalFlip(p=0.5),

        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.1307], std=[0.3081]),
    ])

    # label transform
    def label_transform(label):
        return torch.tensor(label, dtype=torch.int64)

    return image_transform, label_transform


#------------------------------------- split line -------------------------------------#


from datasets import load_dataset
import random


if __name__ == '__main__':
    """code to check preprocess, could run directly"""

    cache_dir = "./.huggingface"
    dataset_path = "mnist"

    # load dataset from huggingface
    mnist_dataset = load_dataset(path=dataset_path, cache_dir=cache_dir)

    print("\nMNIST info:")
    print(mnist_dataset)

    train_data = mnist_dataset['train']
    train_images = train_data['image']
    train_labels = train_data['label']

    # random example in train data
    rdm_idx = random.randint(0, len(train_labels)-1)
    rdm_img = train_images[rdm_idx]
    print(f"\nlabel of index {rdm_idx} example:", train_labels[rdm_idx])
    rdm_img.show()

    # test transform
    image_transform, _ = preprocess_cv()
    out = image_transform(rdm_img)
    print("\noutput type of image_transform:", type(out))
    print("output size of image_transform:", out.size())
