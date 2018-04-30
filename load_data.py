"""
Data loading script for SCAN data.
"""

import collections
import numpy as np
import random
import torch

def load_data(path):
    examples = []
    with open(path) as f:
        for example_id, line in enumerate(f):
            in_ = line.split(":")[1].replace("OUT", "")
            out_ = line.split(":")[2].replace("\n", "")

            example = {}
            example["in_seq"] = in_
            example["in_tokens"] = in_.split()
            example["out_seq"] = out_
            example["out_tokens"] = out_.split()

            examples.append(example)

    return examples

def build_dictionary(dataset):
    """
    Hot mess code. Fix this. Make into class, or at least add an attribute for in vs out.
    """
    in_word_counter = collections.Counter()
    out_word_counter = collections.Counter()
    for example in dataset:
        in_word_counter.update(example["in_tokens"])
        out_word_counter.update(example["out_tokens"])

    in_vocabulary = set([word for word in in_word_counter]) 
    in_vocabulary = list(in_vocabulary)
    in_vocabulary = ["<SOS>", "<EOS>", "<PAD>"] + in_vocabulary

    out_vocabulary = set([word for word in out_word_counter]) 
    out_vocabulary = list(out_vocabulary)
    out_vocabulary = ["<SOS>", "<EOS>", "<PAD>"] + out_vocabulary


    in_word2index = dict(zip(in_vocabulary, range(len(in_vocabulary))))
    in_index2word = {v: k for k, v in in_word2index.items()}

    out_word2index = dict(zip(out_vocabulary, range(len(out_vocabulary))))
    out_index2word = {v: k for k, v in out_word2index.items()}

    for example in dataset:
        in_seq_length = len(example["in_tokens"])
        example["in_index"] = [] #[in_word2index["<SOS>"]] 
        #torch.zeros((seq_length))
        in_sequence = example["in_tokens"]
        for token in in_sequence:
            index = in_word2index[token]
            example["in_index"].append(index)
        example["in_index"].append(in_word2index["<EOS>"])


        out_seq_length = len(example["in_tokens"])
        example["out_index"] = [] #[out_word2index["<SOS>"]]
        out_sequence = example["out_tokens"]
        for token in out_sequence:
            index = out_word2index[token]
            example["out_index"].append(index)
        example["out_index"].append(out_word2index["<EOS>"])

    return (in_word2index, in_index2word, in_vocabulary), (out_word2index, out_index2word, out_vocabulary)

def paddify(batch):
    PAD = 2
    inputs = [batch[i]["in_index"] for i in range(len(batch))]
    in_max_len = len(max(inputs, key=len))

    outs = [batch[i]["out_index"] for i in range(len(batch))]
    out_max_len = len(max(outs, key=len))
    
    for i, example in enumerate(batch):
        in_this_len = len(batch[i]["in_index"])
        batch[i]["in_index"] = batch[i]["in_index"] + [2]*(in_max_len - in_this_len)

        out_this_len = len(batch[i]["out_index"])
        batch[i]["out_index"] = batch[i]["out_index"] + [2]*(out_max_len - out_this_len)

    return batch

# This is the iterator we'll use during training. 
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        batch = [{"in_index": source[index]["in_index"], \
                 "out_index":source[index]["out_index"]} \
                 for index in batch_indices]
        batch = paddify(batch)
        yield batch

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        batch = paddify(batch)
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue
        
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    inputs = []
    outs = []
    for d in batch:
        inputs.append(torch.LongTensor(d["in_index"]).unsqueeze(1))
        outs.append(torch.LongTensor(d["out_index"]).unsqueeze(1))
    return inputs, outs

