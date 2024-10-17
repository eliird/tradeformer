import json
import os
from tqdm import tqdm


def get_all_combinations():
    color = list('abc')
    size_ratio = list('defghij')
    wick_ratio = list('klmnopq')

    opts = [color, size_ratio, wick_ratio]
    res = []

    def fill_opts(i, word=[]):
        if i == 3:
            res.append(word[:])
            return

        for item in opts[i]:
            word += item
            fill_opts(i+1, word)
            word.pop()

    fill_opts(0)
    return res[:]


def get_itos_dicts():

    combos = get_all_combinations()

    stoi = {}
    itos = {}

    for i, c in enumerate(combos):
        # print(i, ''.join(c))
        w = ''.join(c)
        stoi[w] = i
        itos[i] = w
    return (stoi, itos)


stoi, itos = get_itos_dicts()


class MarketTokenizer:

    def __init__(self,
                 load_path: str = None, pad_token: str = '<pad>'):

        self.stoi, self.itos = get_itos_dicts()
        self.unique_tokens = max(self.itos)
        self.merges = {}
        self.vocab = {}
        self.pad_token = pad_token
        if pad_token:
            self.unique_tokens += 1
            self.itos[max(itos) + 1] = pad_token
            self.stoi[self.pad_token] = self.itos[max(itos)]

        if load_path:
            with open(os.path.join(load_path, 'merges.json'), 'r') as f:
                merges = json.load(f)
                self.merges = {tuple(map(int, key.strip("()").split(
                    ", "))): value for key, value in merges.items()}
            with open(os.path.join(load_path, 'vocab.json')) as f:
                self.vocab = json.load(f)
                self.vocab = {int(k): v for k, v in self.vocab.items()}

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def get_stats(self, ids: list) -> dict:
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def encode(self, text: str):
        assert len(self.vocab) > 0
        tokens = [stoi[w.lower()] for w in text.split()]
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        assert len(self.vocab) > 0
        # given ids (list of integers), return candle words
        text = " ".join(self.vocab[idx] for idx in ids)
        return text.upper()

    def train(self, token_data_path: str, vocab_size: int, save_path='./'):
        with open(token_data_path, 'r') as f:
            doc = f.readlines()
        doc = ' '.join(doc).split()
        tokens = [stoi[w.lower()] for w in doc]

        self.merges = {}
        ids = list(tokens)
        num_merges = vocab_size - self.unique_tokens

        for i in tqdm(range(num_merges)):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = self.unique_tokens + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

        self.vocab = dict(self.itos)
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + " " + self.vocab[p1]

        # convert the merges key from tuple to str
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'merges.json')
        merges = {str(key): value for key, value in self.merges.items()}
        with open(save_file_path, 'w') as f:
            json.dump(merges, f, indent=4)

        save_file_path = os.path.join(save_path, 'vocab.json')
        with open(save_file_path, 'w') as f:
            json.dump(self.vocab, f, indent=4)
