import torch
import pandas as pd
import numpy as np
import copy
import json
import os

from util import levenshteinDistanceDP
from loss import FocalTverskyLoss
from arch_classifier import Arch
from seq2seq_translation.seq2seq_gru import Seq2SeqGRU
from sklearn.model_selection import train_test_split

dir_output = os.getcwd()
df = pd.read_pickle('pred_frame_20k.pkl')


df['pred sentence length'] = df['pred sentence'].apply(lambda x: len(x))
df = df[df['pred sentence length'] > 4]

df, df_tst = train_test_split(df, test_size=0.1)

# eos token
EOS = 'μ'
PAD = 'ω'
seq_structure_tokens = [PAD, EOS] 

corpus = copy.deepcopy(seq_structure_tokens)
corpus_set = set(corpus)
sequences = []
sequences_tst = []

max_seq_len = 0
for _, row in df[['sentence', 'pred sentence']].iterrows():
    
    sequence_src, sequence_tgt = (row['pred sentence'], row['sentence'])
    
    while '  ' in sequence_tgt:
    	sequence_tgt.replace('  ', ' ')

    sequence_src.replace(EOS, ' ')
    sequence_tgt.replace(EOS, ' ')
    sequence_src.replace(PAD, ' ')
    sequence_tgt.replace(PAD, ' ')
    sequence_src += EOS 
    sequence_tgt += EOS

    max_seq_len = max((max_seq_len, len(sequence_src), len(sequence_tgt)))
    
    sequences.append((sequence_src, sequence_tgt))

    for sequence in (sequence_src, sequence_tgt):
        for el in sequence:
            if el in seq_structure_tokens:
                continue
            else:
                for token in el:
                    if not token in corpus_set:
                        corpus.append(token)
                        corpus_set.add(token)

for _, row in df_tst[['sentence', 'pred sentence']].iterrows():
    
    sequence_src, sequence_tgt = (row['pred sentence'], row['sentence'])
    
    while '  ' in sequence_tgt:
    	sequence_tgt.replace('  ', ' ')

    sequence_src.replace(EOS, ' ')
    sequence_tgt.replace(EOS, ' ')
    sequence_src.replace(PAD, ' ')
    sequence_tgt.replace(PAD, ' ')
    sequence_src += EOS 
    sequence_tgt += EOS

    max_seq_len = max((max_seq_len, len(sequence_src), len(sequence_tgt)))
    
    sequences_tst.append((sequence_src, sequence_tgt))

    for sequence in (sequence_src, sequence_tgt):
        for el in sequence:
            if el in seq_structure_tokens:
                continue
            else:
                for token in el:
                    if not token in corpus_set:
                        corpus.append(token)
                        corpus_set.add(token)
corpus.append('`')
import pdb; pdb.set_trace()
with open('corpus.csv', 'w') as out:
    for word in corpus:
        out.write(word+'\n')


f = len(corpus)
device = torch.torch.device('cuda:0')


def encode_weights(sentence, enc, weights=[1e-0, 1e+2]):
    sentence_split = sentence.split(" ")
    n = 1
    i = n
    while i < len(sentence_split):
        sentence_split.insert(i, ' ')
        i += (n + 1)

    count = 0
    for i, element in enumerate(sentence_split):
        if element == 'μ':
            enc[0, count, corpus.index(element)] = weights[1]
            count += 1
        else:
            for token in element:
                enc[0, count, corpus.index(token)] = weights[0]
                count += 1
    return enc


def encode_sequence(sentence, enc):
    sentence_split = sentence.split(" ")
    n = 1
    i = n
    while i < len(sentence_split):
        sentence_split.insert(i, ' ')
        i += (n + 1)
    count = 0
    for i, element in enumerate(sentence_split):
        if element in seq_structure_tokens:
            enc[0, count, corpus.index(element)] = 1.0
            count += 1
        else:
            for token in element:
                enc[0, count, corpus.index(token)] = 1.0
                count += 1
    assert count == enc.shape[1]
    return enc

def count_encoding_length(sentence):
    sentence_split = sentence.split(" ")
    count = 0
    n = 1
    i = n
    while i < len(sentence_split):
        sentence_split.insert(i, ' ')
        i += (n + 1)
    for i, element in enumerate(sentence_split):
        if element in seq_structure_tokens:
            count += 1
        else:
            count += len(element)
    return count

def decode_sequence(seq):
    dec = ""
    for vec in seq.argmax(1):
        i = vec.item()
        if corpus[i] == PAD:
           pass
        else:
            dec += corpus[i]
            if corpus[i] == EOS:
                break

    dec = dec.replace('\n', ' ')
    return dec


def is_numeric(string):
    try:
        float(string)
        return True
    except:
        return False
    
    for char in string:
        if not char.isdigit():
            return False
    
    return True


def get_tokens_from_sentence(sentence):
    tokens = set()
    sentence_split = sentence.split(" ")
    for element in sentence_split:
        if element in seq_structure_tokens:
            tokens.add(element)
        elif is_numeric(element):
            continue
        else:
            for token in element:
                if is_numeric(token):
                    continue
                tokens.add(token)
    return list(tokens)

def prepare_sequence(sequence, scale_augment, trace=False):


    # corpus_adjusted = copy.deepcopy(corpus)
    # sequence_tgt_tokens = get_tokens_from_sentence(sequence_tgt)
    
    # if trace:
    #     import pdb; pdb.set_trace()
    # for token in seq_structure_tokens:
    #     corpus_adjusted.remove(token)
    #     if token in sequence_tgt_tokens:
    #         sequence_tgt_tokens.remove(token)

    # skirt around char replace function replacing the 's' in '[eos]' etc.
    # sequence_tgt = sequence_tgt.replace('[eos]', 'δ')
    # sequence_tgt = sequence_tgt.replace('[N]', 'θ')

    # m = 1 + scale_augment // 10000
    # swap_size = min(m, len(sequence_tgt_tokens))
    # swap_size = np.random.choice(np.arange(min(10, len(sequence_tgt_tokens))))
    # # swap_in = np.random.choice(corpus_adjusted, size=swap_size, replace=False)
    # swap_in = ['μ'] * swap_size
    # swap_out = np.random.choice(sequence_tgt_tokens, size=swap_size, replace=False)
    # swap_out = ['t'] * swap_size

    sequence_src, sequence_tgt = sequence

    if np.random.rand() > 0.5:
        size = np.random.choice(np.arange(5))
        while (len(sequence_src) + size) > max_seq_len:
            size -= 1
        if size > 0:
            sequence_src = list(sequence_src)
            insert_symbols = np.random.choice(list(' |.-_!\'"`10O7nmoil,;:') + corpus, replace=True, size=size)
            insert_indices = np.random.choice(np.arange(len(sequence_src)), size=len(insert_symbols))
            for index, symbol in zip(insert_indices, insert_symbols):
                sequence_src.insert(index, symbol)
            sequence_src = ''.join(sequence_src)

    # for (char_out, char_in) in zip(swap_out, swap_in):
    #     sequence_src = sequence_src.replace(char_out, char_in)
        
    # sequence_tgt = sequence_tgt.replace('δ', '[eos]')
    # sequence_tgt = sequence_tgt.replace('θ', '[N]')
    # sequence_src = sequence_src.replace('δ', '[eos]')
    # sequence_src = sequence_src.replace('θ', '[N]')

    n_src = count_encoding_length(sequence_src)    
    n_tgt = count_encoding_length(sequence_tgt)
    
    # batch, seq, enc
    src = torch.zeros((1, n_src, f))
    tgt = torch.zeros((1, n_tgt, f))

    # tokenize
    src = encode_sequence(sequence_src, src)
    tgt = encode_sequence(sequence_tgt, tgt)

    src = torch.cat([src, torch.zeros(1, max_seq_len-src.shape[1], f)], 1)
    tgt = torch.cat([tgt, torch.zeros(1, max_seq_len-tgt.shape[1], f)], 1)

    weights = encode_weights(sequence_src, src)

    return src, tgt, weights

net = Seq2SeqGRU(
    input_dim=len(corpus),
    embed_dim=256,
    hidden_dim=1028,
    output_dim=len(corpus),
    n_layers=(1,1),
    bidirectional=True)

net.to(device)

weight = torch.ones((f))
weight[corpus.index(EOS)] = 1e-2
weight[corpus.index(PAD)] = 1e-3
weight[corpus.index(' ')] = 1e-4
loss = torch.nn.CrossEntropyLoss(reduction='sum', weight=weight).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

c = 0
dist = 0.0

batch_size = 128
log_step = 100
alpha = 1e-1
trn_cost = 0.0

while True:
    
    net.train()

    indices = np.random.choice(np.arange(len(sequences)), size=batch_size)
    srcs = []
    tgts = []
    for ix in indices:
        sequence = sequences[ix]
        src, tgt, weight = prepare_sequence(sequence, scale_augment=c, trace=False)
        srcs.append(torch.Tensor(src))
        tgts.append(torch.Tensor(tgt))
    
    srcs = torch.cat(srcs, 0).float().to(device)
    tgts = torch.cat(tgts, 0).float().to(device)
    
    optimizer.zero_grad()

    out_ix, out = net(srcs, tgts)
    cost = loss(out.permute(0, 2, 1), tgts.argmax(2)) * alpha

    cost.backward()
    optimizer.step()

    trn_cost += cost.detach().cpu().item()

    if (c > 0) and (c % log_step == 0):
        net.eval()

        tst_cost = 0.0
        tst_dist = 0.0

        n = 20 # len(sequences_tst)
        display_ix = np.random.choice(np.arange(n))
        for ix in range(n):
            sequence = sequences_tst[ix]
            src, tgt, weights = prepare_sequence(sequence, scale_augment=c, trace=False)
            srcs = torch.cat([torch.Tensor(src)], 0).float().to(device)
            tgts = torch.cat([torch.Tensor(tgt)], 0).float().to(device)

            with torch.no_grad():
                out_ix, out = net(srcs, tgts, inference=True)
                cost = loss(out.permute(0, 2, 1), tgts.argmax(2)) * alpha
                tst_cost += cost.detach().cpu().item()

            src = srcs[0:1].detach().cpu()[0]
            tgt = tgts[0:1].detach().cpu()[0]
            out = out[0:1].detach().cpu()[0]

            input = decode_sequence(src)
            label = decode_sequence(tgt)
            pred = decode_sequence(out)
            tst_dist += levenshteinDistanceDP(pred, label) / len(label)

            if ix == display_ix:
                display = (input, label, pred)

        trn_cost /= (batch_size*c)
        trn_cost = np.round(trn_cost, 4)
        tst_cost /= n
        tst_cost = np.round(tst_cost, 4)
        tst_dist /= n
        tst_dist = np.round(tst_dist, 4)
        display_row = f"iter: {c} | trn cost: {trn_cost} | tst cost: {tst_cost} | tst dist {tst_dist}"
        
        print(display_row)
        print('-'*len(display_row))
        print(f'input: {display[0]}')
        print('-'*25)
        print(f'pred: {display[2]}')
        print('-'*25)
        print(f'label: {display[1]}')
        print('-'*25)

        state_dict = net.state_dict()
        torch.save(state_dict, os.path.join(dir_output, 'seq2seq-gru-state.pt'))

        trn_cost = 0.0

        if c % (log_step * 5) == 0:
	        for g in optimizer.param_groups:
	            g['lr'] *= 0.7

    c += 1