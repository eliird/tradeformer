import math
import os
import time
import torch
import torch.nn as nn
from dataloader import DataLoaderLite
import torch.nn.functional as F
from tradeformer import (
    TradeFormer,
    TradeFormerConfig,
    MarketTokenizer
)

torch.set_float32_matmul_precision('highest')


def generate(model: TradeFormer,
             tokens: list[str],
             max_len: int = 20, topk: int = 10,
             tokenizer: MarketTokenizer = None, decode=True):
    # TOKENS: B, T
    model.eval()
    while tokens.size(1) < max_len:
        with torch.no_grad():
            out = model(tokens)

            logits = out[:, -1, :]
            probs = F.softmax(logits, 1)

            topk_probs, topk_idx = torch.topk(probs, topk, dim=-1)

            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_idx, -1, ix)  # (B, 1)

            tokens = torch.cat((tokens, xcol), dim=1)
    if decode and tokenizer:
        tokens = [tokenizer.decode(x) for x in tokens.tolist()]
    return tokens


def get_lr(it: int, max_lr=3e-4, min_lr=3e-5, warmup_steps=70, max_steps=50):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    if it > max_steps:
        return min_lr

    decay_raio = (it - warmup_steps) / (max_steps - warmup_steps)

    assert 0 <= decay_raio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_raio))

    return min_lr + coeff * (max_lr - min_lr)


def main():
    # hyperparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    seq_length = 100
    max_steps = 1000
    eval_every = 10
    val_loss_steps = 50
    save_every = 500

    # load model
    model: TradeFormer = TradeFormer(TradeFormerConfig())
    model = model.to(device)
    model = torch.compile(model)
    # optimizer
    # torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer = model.configure_optimizer(
        lr=3e-4, weight_decay=0.1, device=device)

    tokenizer_path = './trained_tokenizer'
    train_data_path = 'Data/preprocessed_data/train.txt'
    val_data_path = 'Data/preprocessed_data/test.txt'
    trainloader = DataLoaderLite(
        tokenizer_path=tokenizer_path,
        data_path=train_data_path,
        B=batch_size,
        T=seq_length,
    )

    valloader = DataLoaderLite(
        tokenizer_path=tokenizer_path,
        data_path=val_data_path,
        B=batch_size,
        T=seq_length,
    )
    # logits, loss = model(x, y)
    for step in range(max_steps):
        model.train()
        batch_time = time.time()
        x, y = trainloader.next_batch()
        x = x.to(device)  # B, T
        y = y.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update the learning rate
        lr = get_lr(step, max_steps=max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        torch.cuda.synchronize()
        dt = time.time() - batch_time
        batch_time = dt * 1000
        tokens_per_sec = (trainloader.B * trainloader.T)/dt

        if step % save_every == 0 and step > 0:
            torch.save(model.state_dict(),
                       f'./model_ckpt/chekpoint_{step}.ckpt')

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = valloader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss/val_loss_steps
                    val_loss_accum += loss.detach()
            print(f'step{step:4d} | loss: {loss.item():.4f} | val loss: {val_loss_accum.item():.4f} | norm: {norm:.4f} | batch_time: {
                batch_time:.2f} ms | throughput : {tokens_per_sec:.2f} tokens/sec'
            )
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
