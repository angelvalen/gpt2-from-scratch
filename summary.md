

# Sentiment Analysis

Expected from instructions:
Last Linear Layer for SST: Dev Accuracy: 0.462
Full Model for SST: Dev Accuracy: 0.513
Last Linear Layer for CFIMDB: Dev Accuracy: 0.861
Full Model for CFIMDB: Dev Accuracy: 0.976

My results:
Last Linear Layer for SST: Dev Accuracy: 0.453
Full Model for SST: Dev Accuracy: 0.506
Last Linear Layer for CFIMDB: Dev Accuracy: 0.878
Full Model for CFIMDB: Dev Accuracy: 0.979

# Paraphrase Detection

Acc: 0.675

# Sonnet Generation

Chrf
Top p sampling: 0.309
Beam search: 0.167

Evaluation time before KV cache
top p: 40 s
beam: 141 s
After
top p: 35 s
beam: 47 s
Huge improvement !


# Initial Baseline Run:
Epochs 50 with 5 patience for all

| Task | Mode | LR | Batch Size | Dropout | Acc/Chrf | Train time | Eval time
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sentiment (SST) | Last linear | 1e-3 | 64 | 0.3 | 0.481 | 1241 | 10
| Sentiment (SST) | Full model | 1e-5 | 32 | 0.1 | 0.504 | 1068 | 9
| Sentiment (CFIMDB) | Last linear | 1e-3 | 32 | 0.3 | 0.869 | 2192 | 23
| Sentiment (CFIMDB) | Full model | 1e-5 | 8 | 0.1 | 0.975 | 1902 | 20
| Paraphrase (28k) | Last linear | 1e-3 | 64 | 0.3 | 0.706 | 2508 | 70
| Paraphrase (28k) | Full model | 1e-5 | 32 | 0.1 | 0.848 | 8022 | 65
| Sonnets | Full model | 1e-5 | 8 | 0.1 |
| Sonnets (decoding) | Top p | temp 0.8–0.9 | top_p 0.9 | - | 0.318 | 446 | 36
| Sonnets (decoding) | Beam search | beams 5 | len penalty 0.6 | - | 0.189 | - | 48
