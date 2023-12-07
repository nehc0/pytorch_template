import torch
# details for torchtext at https://pytorch.org/text/0.16.0/
from torchtext.vocab import build_vocab_from_iterator

from utils import Tokenizer, load_json


def preprocess_nlp(
    train_file: str,
    lowercase: bool = True,
    rm_punctuation: bool = True,
    rm_stopword: bool = False,
    lemmatization: bool = True,
    min_freq: int = 1,
    max_tokens: int = 10000,
):
    """a function for preprocessing in NLP task

    return text_transform, label_transform, vocab, tokenizer
    
    """
    
    # create tokenizer
    tokenizer = Tokenizer(
        lowercase=lowercase, 
        rm_punctuation=rm_punctuation, 
        rm_stopword=rm_stopword, 
        lemmatization=lemmatization,
    )

    # load train data from json
    train_data = load_json(json_file=train_file)

    # generator function to yield tokens from data
    def yield_tokens(data: list[dict]):
        for item in data:
            tokens = tokenizer(item['text'])
            yield tokens

    token_generator = yield_tokens(data=train_data)

    """# for .txt file, load and tokenize line by line
    def yield_tokens(path):
        with open(path, 'r') as f:
            for line in f:
                tokens = tokenizer(line)
                yield tokens
    
    token_generator = yield_tokens(path=train_file)
    """

    # special tokens
    specials = ['<unk>', ]

    # build vocab 
    vocab = build_vocab_from_iterator(
        iterator=token_generator, 
        min_freq=min_freq,  
        max_tokens=max_tokens,
        specials=specials,
        special_first=True,
    )
    # set default index same as index of '<unk>'
    vocab.set_default_index(vocab['<unk>'])

    # text transform
    def text_transform(sentence: str) -> list[torch.int64]:
        # tokenization
        tokens = tokenizer(sentence)
        # vectorization
        long_tensor = torch.tensor(vocab(tokens), dtype=torch.int64)
        return long_tensor

    # label transform, implement it yourself
    def label_transform(label):
        raise NotImplementedError("label_transform function not implemented!")  # TODO

    return text_transform, label_transform, vocab, tokenizer


#------------------------------------- split line -------------------------------------#


from torchvision.transforms import v2


def preprocess_cv():
    """a function for preprocessing in CV task

    return image_transform, label_transform
    
    """

    # image transform
    image_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),

        v2.Resize((128, 128)),
        v2.RandomResizedCrop(size=(128, 128)),
        v2.RandomRotation(degrees=(0, 60)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),

        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # of ImageNet
    ])

    # label transform, implement it yourself
    def label_transform(label):
        raise NotImplementedError("label_transform function not implemented!")  # TODO

    return image_transform, label_transform


#------------------------------------- split line -------------------------------------#


if __name__ == '__main__':
    """code to check preprocess"""

    # check NLP
    example_texts = "./EXAMPLE_TEXTS.json"  # TODO
    test_tokens = ['happy', '<unk>', 'Happy']
    test_indices = [0, 1, 2, 3]
    test_sentence = "Here's a sentence for testing."

    text_transform, _, vocab, tokenizer = preprocess_nlp(example_texts)
    
    print("vocab size: ", len(vocab))
    print(f"indices of {test_tokens}: ", vocab.lookup_indices(test_tokens))
    print(f"tokens of {test_indices}: ", vocab.lookup_tokens(test_indices))
    print(f"sentence \"{test_sentence}\" after tokenization: ", tokenizer(test_sentence))
    print(f"sentence \"{test_sentence}\" after transform: ", text_transform(test_sentence))
    