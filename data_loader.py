from typing import Union

import torch
from datasets import load_dataset, load_from_disk, Dataset as HfDataset
from pyarrow import StringScalar, ChunkedArray
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


class CustomDataset(Dataset):

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], data_path,
                 dataset_name=None,
                 cache_dir: str = None,
                 accept: str = 'train',
                 from_hf: bool = False,
                 max_length: int = 1000,
                 stride: int = 1,
                 max_tokens: int = None,
                 ):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        if self.max_tokens and self.max_tokens < max_length:
            raise ValueError(f'Max tokens {self.max_tokens} is smaller than the max length {max_length}.')

        if max_length < stride:
            raise ValueError(f'Max length {max_length} is larger than stride {stride}.')

        if from_hf:
            self.data = load_dataset(self.data_path, dataset_name, cache_dir=cache_dir)
            self.data: HfDataset = self.data.with_format('torch')
        else:
            self.data: HfDataset = self._load_from_disk()
        self._text = []
        self._input_tokens = []
        self._target_tokens = []

        self._run(accept=accept)

        # We will use sliding window algorith to create input and target batch
        # len(self._text) - max_length this amount of samples created if stride = 1.
        # stride means no of common token prev and next string.
        for index in range(0, len(self._text) - max_length, stride):
            input_tokens = self._text[index:index + max_length]
            target_tokens = self._text[index + 1: index + max_length + 1]
            self._input_tokens.append(torch.tensor(input_tokens))
            self._target_tokens.append(torch.tensor(target_tokens))

    def _load_from_disk(self):
        """
        Generally It is accept DatasetDict only.
        """
        dataset_dict = load_from_disk(self.data_path)
        dataset_dict = dataset_dict.with_format('torch')
        return dataset_dict

    def tokenize(self, text: str):
        return self.tokenizer.encode(text)

    def recursively_collect_text(self, text: str | list[str] | StringScalar, prev: str):
        if isinstance(text, StringScalar):
            return prev + str(text)

        if isinstance(text, ChunkedArray):
            text = text.data

        t = ''
        for sen in text:
            t += self.recursively_collect_text(sen, '\n')
        return t

    def _run(self, accept='train'):
        keys = list(self.data.keys())
        counter = 0
        for key in keys:
            if key in accept:
                value = self.data.pop(key)
                for data in value.data:
                    text = ''
                    break_loop = False
                    if isinstance(data, ChunkedArray):
                        records = data.data
                        for record in records:
                            return_text = self.recursively_collect_text(record, '')
                            counter += len(return_text)
                            text += return_text
                            if self.max_tokens and counter > self.max_tokens:
                                # This loop break for testing model
                                text = self.tokenize(text)
                                self._text.extend(text)
                                break_loop = True
                                break

                    if break_loop:
                        break

                    elif isinstance(data, StringScalar):
                        text = self.recursively_collect_text(data, '')
                        counter += len(text)
                    else:
                        print('Continue Loop because of different type.')
                        continue

                    text = self.tokenize(text)

                    self._text.extend(text)

                    if self.max_tokens and counter > self.max_tokens:
                        # This loop break for testing model
                        break

    def __len__(self):
        return len(self._input_tokens)

    def __getitem__(self, index):
        return self._input_tokens[index], self._target_tokens[index]


def prepare_data(file_path, name, cache_dir, accept, model_name_from_hf, max_length=256, stride=128, max_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_from_hf)
    cd = CustomDataset(tokenizer, file_path, name, cache_dir, accept=accept, from_hf=False,
                       max_length=max_length, stride=stride, max_tokens=max_tokens)

    batches = DataLoader(cd, batch_size=4)

    return iter(batches)

