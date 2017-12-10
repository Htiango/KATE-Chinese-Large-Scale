#coding=UTF-8
import jieba
import io
import json
import os
import re
import string
import numpy as np
from collections import defaultdict

from ..utils.io_utils import dump_json, load_json, write_file
from preprocessing import load_corpus

def load_stopwords(file):
    stop_words = []
    try:
        with open(file, 'r') as f:
            for line in f:
                # print(line);
                stop_words.append(line.strip('\n ').decode('utf-8'))
    except Exception as e:
        raise e
    stop_words_set = set()
    for word in stop_words:
        stop_words_set.add(word)
    return stop_words_set

def init_stopwords():
    try:
        stopword_path = 'patterns/chinese_stopwords.txt'
        cached_stop_words = load_stopwords(os.path.join(os.path.split(__file__)[0], stopword_path))
        print 'Loaded %s' % stopword_path
    except:
        print 'No stopwords dict!'

    return cached_stop_words

def tiny_tokenize(text, cut_all=False, stop_words=set()):
    words = []
    try:
        s_list = jieba.lcut(text, cut_all = False)
        for token in s_list:
            if not token.isdigit() and not token in stop_words :
                words.append(token)
        # print('Finish tokenize!')
        return words
    except Exception as e:
        raise e

# load a doc, determine whether segged
def load_data(corpus_path, segged=True, cut_all=False, stop_words=True):
    word_freq = defaultdict(lambda: 0) # count the number of times a word appears in a corpus
    doc_word_freq = defaultdict(dict) # count the number of times a word appears in a doc

    # word_tokenizer = RegexpTokenizer(r'[a-zA-Z]+') # match only alphabet characters
    # cached_stop_words = init_stopwords()
    cached_stop_words = init_stopwords() if stop_words else set()
    # print('stopwords type: ' )
    # print isinstance(cached_stop_words, list)

    try:
        fp =  open(corpus_path, 'r')
        count_doc = 0;
        while 1:
            lines = fp.readlines()
            if not lines:
                break
            for sentense in lines:
                # print(sentense)
                text = sentense.decode('utf-8').strip('\r\n')
                # print(text)
                # text = sentense
                words = text.split(' ') if segged else tiny_tokenize(text, cut_all, cached_stop_words)
                count_doc += 1
                doc_name = 'line-' + str(count_doc)
                if count_doc % 10000 == 0:
                    print doc_name
                for i in range(len(words)):
                    if not words[i] in cached_stop_words:
                        # doc-word frequency
                        try:
                            doc_word_freq[doc_name][words[i]] += 1
                        except:
                            doc_word_freq[doc_name][words[i]] = 1
                        # word frequency
                        word_freq[words[i]] += 1
    except Exception as e:
        raise e

    return word_freq, doc_word_freq


def construct_corpus(corpus_path, training_phase, vocab_dict=None, threshold=5, topn=None, segged=True):
    if not (training_phase or isinstance(vocab_dict, dict)):
        raise ValueError('vocab_dict must be provided if training_phase is set False')

    word_freq, doc_word_freq = load_data(corpus_path, segged)

    if training_phase:
        vocab_dict = build_vocab(word_freq, threshold=threshold, topn=topn)

    docs = generate_bow(doc_word_freq, vocab_dict)
    new_word_freq = dict([(vocab_dict[word], freq) for word, freq in word_freq.iteritems() if word in vocab_dict])

    return docs, vocab_dict, new_word_freq 

def build_vocab(word_freq, threshold=5, topn=None, start_idx=0):
    """
    threshold only take effects when topn is None.
    words are indexed by overall frequency in the dataset.
    """
    word_freq = sorted(word_freq.iteritems(), key=lambda d:d[1], reverse=True)
    if topn:
        word_freq = zip(*word_freq[:topn])[0]
        vocab_dict = dict(zip(word_freq, range(start_idx, len(word_freq) + start_idx)))
    else:
        idx = start_idx
        vocab_dict = {}
        for word, freq in word_freq:
            if freq < threshold:
                return vocab_dict
            vocab_dict[word] = idx
            idx += 1
    return vocab_dict

def generate_bow(doc_word_freq, vocab_dict):
    docs = {}
    for key, val in doc_word_freq.iteritems():
        word_count = {}
        for word, freq in val.iteritems():
            try:
                word_count[vocab_dict[word]] = freq
            except: # word is not in vocab, i.e., this word should be discarded
                continue
        docs[key] = word_count

    return docs

def construct_train_test_corpus(train_path, test_path, output, threshold=5, topn=None):
    dict_path = 'patterns/medical.txt'
    jieba.load_userdict(os.path.join(os.path.split(__file__)[0], dict_path))

    train_docs, vocab_dict, train_word_freq = construct_corpus(train_path, True, threshold=threshold, topn=topn, segged =True)
    train_corpus = {'docs': train_docs, 'vocab': vocab_dict, 'word_freq': train_word_freq}
    with io.open(os.path.join(output, 'train.corpus'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_corpus, ensure_ascii=False))
    print 'Generated training corpus\n'

    test_docs, _, _ = construct_corpus(test_path, False, vocab_dict=vocab_dict, threshold=threshold, topn=topn, segged=False)
    test_corpus = {'docs': test_docs, 'vocab': vocab_dict}

    with io.open(os.path.join(output, 'test.corpus'), 'w', encoding='utf-8') as f2:
        f2.write(json.dumps(test_corpus, ensure_ascii=False))
    # dump_json(test_corpus, os.path.join(output, 'test.corpus'), ensure_ascii=False)

    print 'Generated test corpus'

    return train_corpus, test_corpus

def construct_train_corpus(train_path, output, threshold=5, topn=None):
    dict_path = 'patterns/medical.txt'
    jieba.load_userdict(os.path.join(os.path.split(__file__)[0], dict_path))

    train_docs, vocab_dict, train_word_freq = construct_corpus(train_path, True, threshold=threshold, topn=topn, segged =False)
    train_corpus = {'docs': train_docs, 'vocab': vocab_dict, 'word_freq': train_word_freq}
    with io.open(os.path.join(output, 'train.corpus'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(train_corpus, ensure_ascii=False))
    print 'Generated training corpus\n'
    return train_corpus

def construct_test_corpus(test_path, vocab_path, output):
    dict_path = 'patterns/medical.txt'
    jieba.load_userdict(os.path.join(os.path.split(__file__)[0], dict_path))
    vocab_dict = load_corpus(vocab_path)
    test_docs, _, _ = construct_corpus(test_path, False, vocab_dict=vocab_dict, topn=None, segged=False)
    test_corpus = {'docs': test_docs, 'vocab': vocab_dict}

    with io.open(os.path.join(output, 'test.corpus'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_corpus, ensure_ascii=False))
    print 'Generated testing corpus\n'
    return test_corpus