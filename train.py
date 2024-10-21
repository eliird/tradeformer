import os
import torch
import torch.nn as nn
from dataloader import DataLoaderLite
import torch.nn.functional as F
from tradeformer import (
    TradeFormer,
    TradeFormerConfig, 
    MarketTokenizer
)



def generate(model: TradeFormer, 
             tokens: list[str], 
             max_len: int=20, topk: int=10,
             tokenizer: MarketTokenizer=None, decode=True):
    # TOKENS: B, T
    model.eval()
    while tokens.size(1) < max_len:
        with torch.no_grad():
            out = model(tokens)

            logits = out[:, -1, :]
            probs = F.softmax(logits, 1)

            topk_probs, topk_idx = torch.topk(probs, topk, dim=-1)

            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_idx, -1, ix) #(B, 1)

            tokens = torch.cat((tokens, xcol), dim=1)
    if decode and tokenizer:
            tokens = [tokenizer.decode(x) for x in tokens.tolist()]
    return tokens


def main():
    # hyperparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_return_sequences = 5
    batch_size = 32
    seq_length = 100
    epochs = 11

    # load model
    model = TradeFormer(TradeFormerConfig())
    model = model.to(device)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    tokenizer_path = './trained_tokenizer'
    data_path = 'Data/preprocessed_data/train.txt'
    dataloader = DataLoaderLite(
        tokenizer_path=tokenizer_path,
        data_path=data_path,
        B = batch_size,
        T = seq_length,
    )
    

    # logits, loss = model(x, y)
    for i in range(50):
        x, y = dataloader.next_batch()    
        x = x.to(device) # B, T
        y = y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f'step{i}, loss: {loss.item()}')
            
    # TO GENERATE
    # print(x.shape)
    # x = generate(model, x, max_len=30, topk=3, tokenizer=tokenizer, decode=True)
    # for item in x:
    #      print(item)



def temp_dataloader(tokenizer: MarketTokenizer, train_data: list[str]):
    # dummy for now
    text = ' '.join(train_data[0].split()[:1000])
    tokens = tokenizer.encode(text)
    B, T = 4, 32
    buf = torch.tensor(tokens[:B*T + 1])
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    return x, y
     
     


if __name__ == "__main__":
    main()
