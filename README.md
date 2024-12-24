# Tradeformer
 -Transformer for trading

## Tokenizer
Built a BPE style tokenizer for markets. The idea is to convert each candle into a character depending on its properties. Below are the properties used to convert a candle into a character. Then run BPE to generate a vocab and merges file.

### Candle to Chars
    ```
    G = Bull Candle = A
    R = Bear Candle = B
    N = No Color Candle = C

    E = Candle == ATR ratio = D
    SSS = Candle <<< ATR = E
    SS = Candle << ATR = F
    S = Candle < ATR = G
    BBB = Candle >>> ATR = H
    BB = Canadle >> ATR = I
    B = Candle > ATR = J

    M = Top Wick == Bot Wick = K
    LLL = Bot Wick >>> TopWick = L
    LL = Bot Wick >> TopWick = M 
    L = Bot Wick > TopWick = N

    UUU = Top Wick >>> Bot Wick = O
    UU = Top Wick >> Bot Wick = P 
    U = Top Wick > Bot Wick = Q

    ```
## Model

The model used is a causal LM (GPT like model). Total number of parameters in 10 million.
