from __future__ import absolute_import
import timeit
import argparse
from os import path
import numpy as np
import io

from autoencoder.preprocessing.preprocessing import load_corpus

from autoencoder.utils.io_utils import dump_json, load_json


def get_word_relationship(args):
    corpus = load_corpus(args.input_corpus)
    doc_vec_dict = corpus['docs']
    vocab_dict = corpus['vocab']
    print 'Load corpus'

    # we have to revort the dict
    dictionary = dict((v,k) for k, v in vocab_dict.iteritems())

    # Here the input top words path is the json file of the label-topwords_ls 
    # should be a dict, each key is a label and its value is the list of top words
    top_words_path = args.input_topwords
    label_topwordls = load_json(top_words_path)
    print 'Load top words of each label'

    label_topwords_vocabnum_dict = {}
    label_topwordindexls_dict = {}
    for label in label_topwordls:
        label_topwords_vocabnum_dict[label] = {}
        topwords_index_ls = []
        for word in label_topwordls[label]:
            topwords_index_ls.append(word)
            label_topwords_vocabnum_dict[label][word] = {}
        label_topwordindexls_dict[label] = topwords_index_ls

    print 'Finish change words into index'

    # in order to save memory and speed it up, I only calculate the word-words frequency of those 
    # in the top word list

    for label in label_topwordindexls_dict:
        print 'Doing label', str(label)
        topwords_idx_set = set(label_topwordindexls_dict[label])

        for filename in doc_vec_dict:
            word_vec_dict = doc_vec_dict[filename]
            result_word_ls = get_word_list(word_vec_dict, topwords_idx_set)
            for word in result_word_ls:
                for doc_word in word_vec_dict:
                    try:
                        label_topwords_vocabnum_dict[label][word][doc_word] += word_vec_dict[doc_word]
                    except:
                        label_topwords_vocabnum_dict[label][word][doc_word] = word_vec_dict[doc_word]

    print 'Finish building the dict of label-topwords-words-num!'

    # now we should get the top of words

    topn = args.topn

    # it is a dict-dict-ls ({label:{words:[top_relative words]}})
    label_topwords_relativewords = {}
    for label in label_topwords_vocabnum_dict:
        label_topwords_relativewords[label] = {}
        for word in label_topwords_vocabnum_dict[label]:
            vocab_num_dict = label_topwords_vocabnum_dict[label][word]
            label_topwords_relativewords[label][word] = sorted(vocab_num_dict,
                key=vocab_num_dict.__getitem__, reverse = True)[:topn]

    print 'Finish sorting the top n word'

    dump_json(label_topwords_relativewords, args.output_json)
    print 'Finish write the json file'

    for label in label_topwords_relativewords:
        filename_o = args.output_dir + 'label-' + str(label) + '.txt'
        print 'filename =' , filename_o
        file_o = open(filename_o, 'w')
        for word_index in label_topwords_relativewords[label]:
            # print 'Is word_index a int:', isinstance(word_index, int)
            text = dictionary[int(word_index)]
            text += ': '
            for top_relative_wordidx in label_topwords_relativewords[label][word_index]:
                text += dictionary[int(top_relative_wordidx)]
                text += ', '
            text += '\n'
            file_o.write(text.encode('utf-8'))
        file_o.close()
    print 'Finish writing files!'


                
def get_word_list(word_vec_dict, vocab_set):
    result_word_ls = []
    for word in word_vec_dict:
        if word in vocab_set:
            result_word_ls.append(word)
    return result_word_ls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ic', '--input_corpus', type=str, required = True, 
        help = 'path of the input filename corpus dict')
    parser.add_argument('-il', '--input_topwords', type=str, required = True, 
        help = 'path of the json input label-word dict')
    parser.add_argument('-tn', '--topn', type=int, required = True, 
        help = 'number of top words of a cluster')
    parser.add_argument('-oj', '--output_json', type = str, required = True, 
        help='path of the output json file')
    parser.add_argument('-o', '--output_dir', type=str, required = True, 
        help = 'dir of the output top n relative words of each label file')
    args = parser.parse_args()
    get_word_relationship(args)


if __name__ == '__main__':
    main()