import os
import torch
import pickle
from tqdm import tqdm
from tradeformer import MarketTokenizer


class DataLoaderLite:
    def __init__(self, tokenizer_path, data_path: str, B, T, force_create=False):
        self.tokenizer = MarketTokenizer(tokenizer_path)
        self.B = B
        self.T = T

        with open(data_path, 'r') as f:
            text = f.readlines()

        cached_tokens_path = data_path.split('/')[-1].replace('.txt', '.pkl')

        if os.path.exists(cached_tokens_path) and not force_create:
            print(
                "loading cached tokens for the dataset, Use force create argument to recreate tokens")
            with open(cached_tokens_path, 'rb') as f:
                self.tokens = pickle.load(f)
        else:
            print("creating and caching tokens")
            tokens = [self.tokenizer.encode(x) for x in tqdm(text)]
            self.tokens = torch.tensor(tokens)
            with open(cached_tokens_path, 'wb') as f:
                pickle.dump(self.tokens, f)

        self.len = self.tokens.shape[0] * self.tokens.shape[1]
        print(f"loaded {self.len} tokens")
        print(f"1 epoch = {self.len // (B * T)} batches")

        # state
        self.current_file = 0
        self.current_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.get_buffer(B, T)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B * T
        if self.current_pos + (B*T + 1) > len(self.tokens[self.current_file]):
            self.current_file = (self.current_file + 1) % self.tokens.shape[0]
            self.current_pos = 0
        return (x, y)

    def random_batch(self):
        B, T = self.B, self.T
        file = torch.randint(0, self.tokens.shape[0], (1,)).item()
        pos = torch.randint(0, self.tokens.shape[1] - (B * T) - 1, (1,)).item()
        buf = self.tokens[file][pos: pos+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        return (x, y)

    def get_buffer(self, B, T):
        return self.tokens[self.current_file][self.current_pos: self.current_pos+B*T+1]
