import torch
# details for torchtext at https://pytorch.org/text/0.16.0/
from torchtext.vocab import build_vocab_from_iterator

from utils import Tokenizer


def preprocess_nlp(
    train_text: list[str],
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

    # yield tokens
    def yield_tokens(data: list[str]):
        for sentence in data:
            tokens = tokenizer(sentence)
            yield tokens
    
    token_generator = yield_tokens(data=train_text)

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

    # label transform
    def label_transform(label: int) -> torch.int64:
        return torch.tensor(label, dtype=torch.int64)

    return text_transform, label_transform, vocab, tokenizer


#------------------------------------- split line -------------------------------------#


from datasets import load_dataset
import random


if __name__ == '__main__':
    """code to check preprocess, could run directly"""

    # load data from huggingface
    cache_dir = "./.huggingface"
    dataset_path = "imdb"
    # only use a subset to save time
    train_data = load_dataset(path=dataset_path, cache_dir=cache_dir, split="train[:1000]") 
    print("\ntrain data info:")
    print(train_data)

    # example in train data
    train_text = train_data['text']
    train_label = train_data['label']
    rdm_idx = random.randint(0, len(train_label)-1)
    print("\nindex:", rdm_idx)
    print("text:", train_text[rdm_idx])
    print("label:", train_label[rdm_idx])

    text_transform, _, vocab, tokenizer = preprocess_nlp(train_text=train_text)

    # test preprocess
    test_tokens = ['happy', '<unk>', 'Happy', 'good', 'bad']
    test_indices = [0, 1, 2, 3, 4, 5]
    test_sentence = "This's a really good movie."
    print("\nvocab size: ", len(vocab))
    print(f"indices of {test_tokens}: ", vocab.lookup_indices(test_tokens))
    print(f"tokens of {test_indices}: ", vocab.lookup_tokens(test_indices))
    print(f"sentence \"{test_sentence}\" after tokenization: ", tokenizer(test_sentence))
    print(f"sentence \"{test_sentence}\" after transform: ", text_transform(test_sentence))
    