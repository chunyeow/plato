import os
import torch
import hashlib
from tqdm import tqdm
import gzip
from abc import abstractmethod
from torch.utils.data import Dataset
from six.moves import urllib
import numpy as np
import tarfile
import errno
import zipfile

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    def __init__(self):
        super().__init__()
        _path = Config().params["data_path"]
        self.trainset = BatchDataset(WikiText2(root=_path, split="train"), 64)
        self.testset = BatchDataset(WikiText2(root=_path, split="valid"), 64)

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)


class BatchDataset(Dataset):
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = len(dataset)
        self.idx = list(range(0, self.S, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        if self.S - self.idx[index] >= self.seq_length:
            seq_length = self.seq_length
            input = {
                "label": self.dataset[self.idx[index] : self.idx[index] + seq_length][
                    "label"
                ]
            }
        else:
            sequence = self.dataset[
                self.idx[index] - self.seq_length : self.idx[index]
            ]["label"]
            input = {"label": sequence}
        return input


def calculate_md5(path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(path, md5, **kwargs):
    return md5 == calculate_md5(path, **kwargs)


def check_integrity(path, md5=None):
    if not os.path.isfile(path):
        return False
    if md5 is None:
        return True
    return check_md5(path, md5)


def make_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, protocol=2, mode="torch"):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == "torch":
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == "numpy":
        np.save(path, input, allow_pickle=True)
    else:
        raise ValueError("Not valid save mode")
    return


def load(path, mode="torch"):
    if mode == "torch":
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == "numpy":
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError("Not valid save mode")
    return


def download_url(url, root, filename, md5):
    path = os.path.join(root, filename)
    makedir_exist_ok(root)
    if os.path.isfile(path) and check_integrity(path, md5):
        print("Using downloaded and verified file: " + path)
    else:
        try:
            print("Downloading " + url + " to " + path)
            urllib.request.urlretrieve(
                url, path, reporthook=make_bar_updater(tqdm(unit="B", unit_scale=True))
            )
        except OSError:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + path
                )
                urllib.request.urlretrieve(
                    url,
                    path,
                    reporthook=make_bar_updater(tqdm(unit="B", unit_scale=True)),
                )
        if not check_integrity(path, md5):
            raise RuntimeError("Not valid downloaded file")
    return


def extract_file(src, dest=None, delete=False):
    print("Extracting {}".format(src))
    dest = os.path.dirname(src) if dest is None else dest
    filename = os.path.basename(src)
    if filename.endswith(".zip"):
        with zipfile.ZipFile(src, "r") as zip_f:
            zip_f.extractall(dest)
    elif filename.endswith(".tar"):
        with tarfile.open(src) as tar_f:
            tar_f.extractall(dest)
    elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        with tarfile.open(src, "r:gz") as tar_f:
            tar_f.extractall(dest)
    elif filename.endswith(".gz"):
        with open(src.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(src) as zip_f:
            out_f.write(zip_f.read())
    if delete:
        os.remove(src)
    return


class Vocab:
    def __init__(self):
        self.symbol_to_index = {"<ukn>": 0, "<eos>": 1}
        self.index_to_symbol = ["<ukn>", "<eos>"]

    def add(self, symbol):
        if symbol not in self.symbol_to_index:
            self.index_to_symbol.append(symbol)
            self.symbol_to_index[symbol] = len(self.index_to_symbol) - 1
        return

    def delete(self, symbol):
        if symbol in self.symbol_to_index:
            self.index_to_symbol.remove(symbol)
            self.symbol_to_index.pop(symbol, None)
        return

    def __len__(self):
        return len(self.index_to_symbol)

    def __getitem__(self, input):
        if isinstance(input, int):
            if len(self.index_to_symbol) > input >= 0:
                output = self.index_to_symbol[input]
            else:
                output = "<ukn>"
        elif isinstance(input, str):
            if input not in self.symbol_to_index:
                output = self.symbol_to_index["<ukn>"]
            else:
                output = self.symbol_to_index[input]
        else:
            raise ValueError("Not valid data type")
        return output

    def __contains__(self, input):
        if isinstance(input, int):
            exist = len(self.index_to_symbol) > input >= 0
        elif isinstance(input, str):
            exist = input in self.symbol_to_index
        else:
            raise ValueError("Not valid data type")
        return exist


class LanguageModeling(Dataset):
    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.token = load(os.path.join(self.processed_folder, "{}.pt".format(split)))
        self.vocab = load(os.path.join(self.processed_folder, "meta.pt".format(split)))

    def __getitem__(self, index):
        input = {"label": self.token[index]}
        return input

    def __len__(self):
        return len(self.token)

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    def _check_exists(self):
        return os.path.exists(self.processed_folder)

    @abstractmethod
    def process(self):
        raise NotImplementedError

    @abstractmethod
    def download(self):
        raise NotImplementedError

    def __repr__(self):
        fmt_str = "Dataset {}\nRoot: {}\nSplit: {}".format(
            self.__class__.__name__, self.root, self.split
        )
        return fmt_str


class PennTreebank(LanguageModeling):
    data_name = "PennTreebank"
    file = [
        (
            "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
            None,
        ),
        (
            "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
            None,
        ),
        (
            "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
            None,
        ),
    ]

    def __init__(self, root, split):
        super().__init__(root, split)

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, "train.pt"))
        save(valid_set, os.path.join(self.processed_folder, "valid.pt"))
        save(test_set, os.path.join(self.processed_folder, "test.pt"))
        save(meta, os.path.join(self.processed_folder, "meta.pt"))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for url, md5 in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def make_data(self):
        vocab = Vocab()
        read_token(vocab, os.path.join(self.raw_folder, "ptb.train.txt"))
        read_token(vocab, os.path.join(self.raw_folder, "ptb.valid.txt"))
        train_token = make_token(vocab, os.path.join(self.raw_folder, "ptb.train.txt"))
        valid_token = make_token(vocab, os.path.join(self.raw_folder, "ptb.valid.txt"))
        test_token = make_token(vocab, os.path.join(self.raw_folder, "ptb.test.txt"))
        return train_token, valid_token, test_token, vocab


class WikiText2(LanguageModeling):
    data_name = "WikiText2"
    file = [
        (
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
            None,
        )
    ]

    def __init__(self, root, split):
        super().__init__(root, split)

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, "train.pt"))
        save(valid_set, os.path.join(self.processed_folder, "valid.pt"))
        save(test_set, os.path.join(self.processed_folder, "test.pt"))
        save(meta, os.path.join(self.processed_folder, "meta.pt"))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for url, md5 in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def make_data(self):
        vocab = Vocab()
        read_token(
            vocab, os.path.join(self.raw_folder, "wikitext-2", "wiki.train.tokens")
        )
        read_token(
            vocab, os.path.join(self.raw_folder, "wikitext-2", "wiki.train.tokens")
        )
        train_token = make_token(
            vocab, os.path.join(self.raw_folder, "wikitext-2", "wiki.train.tokens")
        )
        valid_token = make_token(
            vocab, os.path.join(self.raw_folder, "wikitext-2", "wiki.valid.tokens")
        )
        test_token = make_token(
            vocab, os.path.join(self.raw_folder, "wikitext-2", "wiki.test.tokens")
        )
        return train_token, valid_token, test_token, vocab


class WikiText103(LanguageModeling):
    data_name = "WikiText103"
    file = [
        (
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
            None,
        )
    ]

    def __init__(self, root, split):
        super().__init__(root, split)

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, "train.pt"))
        save(valid_set, os.path.join(self.processed_folder, "valid.pt"))
        save(test_set, os.path.join(self.processed_folder, "test.pt"))
        save(meta, os.path.join(self.processed_folder, "meta.pt"))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for url, md5 in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def make_data(self):
        vocab = Vocab()
        read_token(
            vocab, os.path.join(self.raw_folder, "wikitext-103", "wiki.train.tokens")
        )
        read_token(
            vocab, os.path.join(self.raw_folder, "wikitext-103", "wiki.train.tokens")
        )
        train_token = make_token(
            vocab, os.path.join(self.raw_folder, "wikitext-103", "wiki.train.tokens")
        )
        valid_token = make_token(
            vocab, os.path.join(self.raw_folder, "wikitext-103", "wiki.valid.tokens")
        )
        test_token = make_token(
            vocab, os.path.join(self.raw_folder, "wikitext-103", "wiki.test.tokens")
        )
        return train_token, valid_token, test_token, vocab


def read_token(vocab, token_path):
    with open(token_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split() + ["<eos>"]
            for symbol in line:
                vocab.add(symbol)
    return


def make_token(vocab, token_path):
    token = []
    with open(token_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split() + ["<eos>"]
            for symbol in line:
                token.append(vocab[symbol])
    token = torch.tensor(token, dtype=torch.long)
    return token
