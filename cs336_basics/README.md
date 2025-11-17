## 2.5 Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
It takes 37 hours of training. RAM is 37GB.
The longest token is b"c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382c383c382", token ID is 25851, stands for "ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ", which is not meaningful.

## 2.7 Experiments

### (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.

Here is the results:
||Tiny Text|owt text|
|---|---|---|
|Tiny Tok|4.05|3.50|
|owt Tok|3.93|4.44|

The compression ratio drops.
- For Tiny tokenizer, the corpus contains plain English, the trained tokenizer cannot merge non-Enlighs tokens.
- For OWT tokenizer, the corpus contains multi-lingual and noisy web data, tokenizer may overfit into some weird bytes patterns.

### (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
- Thoughput of Tokenizer is 1.2 MB/sec. It take OWT  Tokenizer 200 hours to tokenize the Pile dataset, 3 hours to tokenize owt_train, 0.5 hours to tokenize TinyStoriesV2-GPT4-train.

### (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?

Vocabulary is 10,000 and 32,000 which fit uint16 range [0, 65535], as uint8 is too small and uint32 is too large.


### Problem (learning_rate_tuning): Tuning the learning rate (1 point)
- When settting small learning rate, loss converge much slower
- When setting large learning rate, loss may expload rather than converage.
- Set appropriate learning rate is crucial for fast and stable training loop.