"""utility functions and classes for pytorch

* def collate_fn(data: list[tuple[torch.Tensor, torch.Tensor]])
* def setup_seed(seed: int) -> None
* class Tokenizer:
    def __init__(
        self, 
        lowercase: bool = False, 
        rm_punctuation: bool = False, 
        rm_stopword: bool = False, 
        lemmatization: bool = False,
    )
    def __call__(self, sentence: str) -> list[str]
* def estimate_model_size(model: torch.nn.Module) -> None
* def load_json(json_file: str)
* def plot_lr_scheduler(
        scheduler_name: str,
        max_epoch: int = 100,
        lr: float = 0.1,
        **kwargs_for_scheduler,
    ) -> None
* def dict_to_str(d: dict) -> str
* def batch_accuracy_cnt(logits: torch.Tensor, labels: torch.Tensor) -> int

"""


#------------------------------------- split line -------------------------------------#


import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(data: list[tuple[torch.Tensor, torch.Tensor]]):
    tensors, targets = zip(*data)
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    return features, targets


#------------------------------------- split line -------------------------------------#


import torch
import numpy as np
import random


def setup_seed(seed: int) -> None:
    """set up seed for reproducibility
    
    details at https://pytorch.org/docs/stable/notes/randomness.html

    """
    assert type(seed) is int, f"seed: type `int` expected, got `{type(seed)}`"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # default: False, cause cuDNN not to benchmark multiple convolution algorithms
    torch.backends.cudnn.benchmark = False
    # default: False, cause cuDNN to only use deterministic convolution algorithms
    torch.backends.cudnn.deterministic = True


#------------------------------------- split line -------------------------------------#


import spacy


class Tokenizer:
    """a class to create tokenizer for nlp task

    tokenizer receive a sentence(str), return tokens(list[str])
    
    """
    def __init__(
        self, 
        lowercase: bool = False, 
        rm_punctuation: bool = False, 
        rm_stopword: bool = False, 
        lemmatization: bool = False,
    ):
        self.lowercase = lowercase
        self.rm_punctuation = rm_punctuation
        self.rm_stopword = rm_stopword
        self.lemmatization = lemmatization
        self.spacy_en = spacy.load('en_core_web_sm')

    def __call__(self, sentence: str) -> list[str]:
        sentence = sentence.strip()
        if self.lowercase is True:
            sentence = sentence.lower()

        tokens = []
        for tok in self.spacy_en(sentence):
            if self.rm_punctuation is True and tok.is_punct:
                continue
            if self.rm_stopword is True and tok.is_stop:
                continue
            if self.lemmatization is True:
                tokens.append(tok.lemma_)
            else:
                tokens.append(tok.text)

        return tokens


"""# an example
if __name__ == '__main__':
    tokenizer = Tokenizer(
        lowercase=True,
        rm_punctuation=True,
        rm_stopword=True,
        lemmatization=True,
    )
    test_texts = "OpenAI's GPT-4 Turbo, with an extensive 128k context window, "\
        "fails to revolutionise as expected due to the \"Lost in the Middle\" phenomenon, "\
        "impacting information recall accuracy. "
    tokens = tokenizer(test_texts)
    print(tokens)
"""


#------------------------------------- split line -------------------------------------#


import torch


def estimate_model_size(model: torch.nn.Module) -> None:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('estimated model size: {:.3f}MB'.format(size_all_mb))


#------------------------------------- split line -------------------------------------#


import json


def load_json(json_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


#------------------------------------- split line -------------------------------------#


import torch
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt


def plot_lr_scheduler(
    scheduler_name: str,  # the name of the pytorch lr scheduler
    max_epoch: int = 100,  # max epoch
    lr: float = 0.1,  # initial lr
    **kwargs_for_scheduler,  # will be passed to scheduler
) -> None:
    """plot the lr change under given schedule method

    you are supposed to pass the kwargs that your scheduler needs

    there are some more schedulers that pytorch provide but not include here
    you can add those into `scheduler_dict` yourself

    """
    # some(not all) lr scheduler that pytorch provide
    scheduler_dict = {
        "LambdaLR": lr_scheduler.LambdaLR,
        "StepLR": lr_scheduler.StepLR,
        "MultiStepLR": lr_scheduler.MultiStepLR,
        "ExponentialLR": lr_scheduler.ExponentialLR,
        "CyclicLR": lr_scheduler.CyclicLR,
        "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
        "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    }

    if scheduler_name not in scheduler_dict.keys():
        print(f"Scheduler '{scheduler_name}' not found. "
            f"These are available: {scheduler_dict.keys()}.")

    # just for testing
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = scheduler_dict[scheduler_name](optimizer, **kwargs_for_scheduler)

    lr_list = []
    for epoch in range(max_epoch):
        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]
        lr_list.append(last_lr)
    x_list = list(range(max_epoch))
    plt.plot(x_list, lr_list)
    plt.show()


"""# an example
if __name__ == '__main__':
    plot_lr_scheduler(
        scheduler_name="CosineAnnealingWarmRestarts",
        T_0=5,
        T_mult=2,
    )
"""


#------------------------------------- split line -------------------------------------#


def dict_to_str(d: dict) -> str:
    out = ""
    for k, v in d.items():
        if type(v) is dict:
            out += str(k) + ": {\n" + dict_to_str(v) + "}\n"
        else:
            out += str(k) + ": " + str(v) + "\n"
    return out


"""# an example
if __name__ == '__main__':
    test_dict = {'a': 1, 'b': {'c': 'hello', 'd': 'world'}, 'e': None}
    dict_str = dict_to_str(test_dict)
    print(dict_str)
"""


#------------------------------------- split line -------------------------------------#


import torch


def batch_accuracy_cnt(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """help to compute accuracy

    given batch of logits(model outputs) and labels
    return the counts of accurate predict
    
    """
    predicts = torch.argmax(logits, dim=-1)
    count = sum(predicts == labels)
    return count.item()


"""# an example
if __name__ == '__main__':
    logits = [[0.6, 0.4],
              [0.3, 0.7],
              [0.8, 0.2]]
    labels = [0, 1, 1]
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    count = batch_accuracy_cnt(logits=logits, labels=labels)
    print(count)
"""


#------------------------------------- split line -------------------------------------#
