

## Train Language Model

Parameters other than those specified will be randomized.


RNN CRNN:

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-rnn.py --language german  --save-to wiki-german-nospaces-bptt-rnn-MYID
```

LSTM CRNN:

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2.py --language english  --save-to wiki-english-nospaces-bptt-MYID --hidden_dim 1024
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2.py --language german  --save-to wiki-german-nospaces-bptt-MYID --hidden_dim 1024
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2.py --language italian  --save-to wiki-italian-nospaces-bptt-MYID --hidden_dim 1024
```

WNLM:

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words.py --language german  --save-to wiki-german-nospaces-bptt-rnn-MYID --hidden_dim 1024 --layer_num 2 
```

Control study: CRNNs with whitespace:
```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-WHITESPACE.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --lr_decay 0.95 --save-to wiki-english-nospaces-bptt-WHITESPACE-MYID --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-WHITESPACE.py --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language german --layer_num 2 --learning_rate 2.0 --lr_decay 0.7 --save_to wiki-german-nospaces-bptt-WHITESPACE-MYID --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.01

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-WHITESPACE.py --batchSize 128 --char_dropout_prob 0.0 --char_embedding_size 100 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 3 --learning_rate 3.5 --lr_decay 0.98 --save_to wiki-italian-nospaces-bptt-WHITESPACE-MYID --sequence_length 80 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.0
```


# train autoencoder baseline
```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-encoderbaseline.py --language german --save-to wiki-autoencoder --batchSize 32

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-encoderbaseline.py --language italian --save-to wiki-autoencoder-italian --batchSize 32
```




## Clustering Character Embeddings


## Phonotactics
Training:
```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-two-sequences-1epoch.py --language german  --save-to wiki-german-nospaces-bptt-SEQUENCE-ITERATIONS --hidden_dim 1024 --layer_num 2 --learning_rate 2.0 --sequences bu,bt
```

Testing:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-two-sequences-start.py --language german --hidden_dim 1024 --layer_num 2 --learning_rate 2.0 --sequences bu,bt --iterations 1epoch
```



## Segmentation

### LSTM CRNN
```
python lm-wiki-segmentation-tokenized.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python lm-wiki-segmentation-tokenized.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412

python lm-wiki-segmentation-tokenized.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --lr_decay 0.95 --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0
```

### RNN CRNN

```
python lm-wiki-segmentation-tokenized-rnn.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083 --nonlinearity tanh

python lm-wiki-segmentation-tokenized-rnn.py --batchSize 256 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language italian --layer_num 2 --learning_rate 0.004 --lr_decay 0.98 --nonlinearity tanh --load-from wiki-italian-nospaces-bptt-rnn-557654324 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0

python lm-wiki-segmentation-tokenized-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --nonlinearity relu --load-from wiki-english-nospaces-bptt-rnn-891035072 --sequence_length 50 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```

### Return profiles around boundaries

```
python lm-wiki-segmentation-tokenized-profile.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python lm-wiki-segmentation-tokenized-profile-flat.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python lm-wiki-segmentation-tokenized-profile-flat.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412

python lm-wiki-segmentation-tokenized-profile-flat.py  --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --lr_decay 0.95 --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0

python lm-wiki-segmentation-tokenized-profile-flat-ci.py  --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --lr_decay 0.95 --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0
```

### Ablation


```
python lm-wiki-segmentation-tokenized-ablation.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python lm-wiki-segmentation-tokenized-ablation.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412

python lm-wiki-segmentation-tokenized-ablation.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --lr_decay 0.95 --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0
```






### Word boundary classification

```
python char-lm-ud-wiki-classify-boundaries.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412
```

```
python char-lm-ud-wiki-classify-boundaries-baseline.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412
```







## POS classification
LSTM CRNN:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS.py --language italian --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412
```

RNN CRNN:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS-rnn.py --batchSize 256 --char_dropout_prob 0.01 --char_embedding_size 50 --char_noise_prob 0.0 --hidden_dim 2048 --language german --layer_num 2 --learning_rate 0.1 --lr_decay 0.95 --nonlinearity tanh --load-from wiki-german-nospaces-bptt-rnn-237671415 --sequence_length 30 --verbose True --weight_dropout_hidden 0.0 --weight_dropout_in 0.0 --train_size 20

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS-rnn.py --batchSize 256 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language italian --layer_num 2 --learning_rate 0.004 --lr_decay 0.98 --nonlinearity tanh --load-from wiki-italian-nospaces-bptt-rnn-557654324 --sequence_length 20 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0 --train_size 20
```

WNLM:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-embeddings.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-embeddings.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 2 --learning_rate 1.2 --lr_decay 0.98 --load-from wiki-italian-nospaces-bptt-words-316412710 --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.0
```

WNLM, including OOVs:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-embeddings-withOOV.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-embeddings-withOOV.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language italian --layer_num 2 --learning_rate 1.2 --lr_decay 0.98 --load-from wiki-italian-nospaces-bptt-words-316412710 --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.0
```

Autoencoder:
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-autoencoder

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-POS.py --language italian --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-autoencoder-italian
```






## Plurals
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-baseline.py --language German --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from-baseline german-autoencoder-var-words
```

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```

## Morphosyntax

### Gender

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-WORDS.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-RNN.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083
```


restrict to Word LSTM vocabulary
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-LEXICON.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-gender-RNN-LEXICON.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083
```


### Case agreement

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-WORDS.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-RNN.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-LEXICON.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-art-adj-noun-cleaned-RNN-LEXICON.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083
```


### Subcategorization

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-prep-noun-cleaned.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-prep-noun-cleaned-WORDS.py --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-prep-noun-cleaned-RNN.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-prep-noun-cleaned-RNN-LEXICON.py --language german --batchSize 128 --char_embedding_size 50 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-rnn-52168083

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-prep-noun-cleaned-LEXICON.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```



## Phonotactics


## Segmentation

```
python lm-wiki-segmentation-tokenized-analysis.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909 --sequence_length 50

python lm-wiki-segmentation-tokenized-analysis.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --lr_decay 0.95 --myID 282506230 --load-from wiki-english-nospaces-bptt-282506230 --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0
```

N-Gram Baseline
```
python lm-wiki-segmentation-tokenized-analysis-ngrams.py --language german 

python lm-wiki-segmentation-tokenized-analysis-ngrams.py --language italian 

python lm-wiki-segmentation-tokenized-analysis-ngrams.py --language english
```


Syntactic Height

```
python lm-wiki-segmentation-tokenized-syntax.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```


# plural classification

## Baseline
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-baseline.py --language German --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from-baseline german-autoencoder-var-words
```

## CRNN
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-autoencoder
```

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-tests.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-tests.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-autoencoder
```

## Vanilla CRNN
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-tests-RNN.py  --batchSize 256 --char_dropout_prob 0.01 --char_embedding_size 50 --char_noise_prob 0.0 --hidden_dim 2048 --language german --layer_num 2 --learning_rate 0.1 --nonlinearity tanh --load-from wiki-german-nospaces-bptt-rnn-237671415 --sequence_length 30 --weight_dropout_hidden 0.0 --weight_dropout_in 0.0
```

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-tests-words.py  --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846
```


```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-tests-words-withOOV.py  --language german --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-words-966024846
```




# for inspecting the data set
```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2-tests-look_at_data.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```





```
python char-lm-ud-stationary-vocab-wiki-nospaces-embeddings-clustering-output.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```



```
python char-lm-ud-stationary-vocab-wiki-nospaces-embeddings-clustering-output.py --language italian --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412
```


```
python char-lm-ud-stationary-vocab-wiki-nospaces-embeddings-tsne-output.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```


```
python char-lm-ud-stationary-vocab-wiki-nospaces-embeddings-clustering-output-phonetic.py --language german --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.35 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-german-nospaces-bptt-910515909
```

```
python char-lm-ud-stationary-vocab-wiki-nospaces-embeddings-clustering-output-phonetic.py --language italian --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 2 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-italian-nospaces-bptt-855947412
```

```
python char-lm-ud-stationary-vocab-wiki-nospaces-embeddings-clustering-output-phonetic.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 
```

```
python char-lm-ud-stationary-vocab-wiki-nospaces-embeddings-clustering-output.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 
```





```
python lm-surprisal-eval-2-childes-ablation.py —load-from initial-childes
```





# Italian Tests

```
python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-adv_aoadj.py --language italian --batchSize 512 --char_dropout_prob 0.0 --char_embedding_size 100 --char_noise_prob 0.0 --hidden_dim 1024  --layer_num 3 --learning_rate 2.0 --load-from wiki-italian-nospaces-bptt-887669069 --sequence_length 50 --weight_dropout_hidden 0.01 --weight_dropout_in 0.0

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-adv_aoadj_RNN.py --language italian --batchSize 512 --char_dropout_prob 0.0 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048  --layer_num 2 --learning_rate 2.0 --load-from wiki-italian-nospaces-bptt-rnn-557654324 --sequence_length 50 --weight_dropout_hidden 0.0 --weight_dropout_in 0.01

python char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-forms-newtests-adv_aoadj_WORDS.py --language italian --batchSize 512 --char_dropout_prob 0.0 --char_embedding_size 300 --char_noise_prob 0.0 --hidden_dim 1024  --layer_num 2 --learning_rate 2.0 --load-from wiki-german-nospaces-bptt-words-20176990 --sequence_length 50 --weight_dropout_hidden 0.0 --weight_dropout_in 0.01
```




# MSR Sentence Completion


## Word RNN


```
python char-lm-ud-stationary-completion-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from wiki-english-nospaces-bptt-words-805035971 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0

python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from wiki-english-nospaces-bptt-words-805035971 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0 --save-to holmes-words-from-805035971-MYID

python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0 --save-to holmes-words-from-fresh-MYID

python char-lm-ud-stationary-completion-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from holmes-words-from-805035971-218115572 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0

python char-lm-ud-stationary-completion-words.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from holmes-words-from-fresh-77128193 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0

python char-lm-ud-stationary-completion-words-VOCAB.py --batchSize 128 --char_dropout_prob 0.01 --char_embedding_size 1024 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 2 --learning_rate 1.1 --lr_decay 1.0 --load-from holmes-words-from-fresh-vocab-237230358 --sequence_length 50 --verbose True --weight_dropout_hidden 0.15 --weight_dropout_in 0.0
```







```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes.py --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.5 --lr_decay 0.95  --load-from wiki-english-nospaces-bptt-282506230 --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0 --save-to holmes-from-282506230-MYID
```

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --save-to holmes-rnn-from-891035072-MYID --lr_decay 0.9 --nonlinearity relu --load-from wiki-english-nospaces-bptt-rnn-891035072 --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```

```
python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-holmes-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --save-to holmes-rnn-from-fresh-MYID --lr_decay 0.9 --nonlinearity relu --sequence_length 50 --verbose True --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```




```
python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from holmes-from-282506230-684739660
```

```
python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from holmes-from-282506230-932742390
```

```
python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 100 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from holmes-from-fresh-754593543
```


```
python char-lm-ud-stationary-completion-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --nonlinearity relu --load-from holmes-rnn-from-fresh-388239891 --sequence_length 50 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```

```
python char-lm-ud-stationary-completion-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --nonlinearity relu --load-from holmes-rnn-from-891035072-134123184 --sequence_length 50 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```

```
python char-lm-ud-stationary-completion-rnn.py --batchSize 256 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 2048 --language english --layer_num 2 --learning_rate 0.01 --nonlinearity relu --load-from wiki-english-nospaces-bptt-rnn-891035072 --sequence_length 50 --weight_dropout_hidden 0.05 --weight_dropout_in 0.01
```




# Word Vectors and Probing Tasks

```
python char-lm-ud-stationary-wordVectors.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 
```

```
python char-lm-ud-stationary-wordVectors-pool.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 
```

```
python char-lm-ud-stationary-probing.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 
```

```
python char-lm-ud-stationary-probing-pool.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 
```



