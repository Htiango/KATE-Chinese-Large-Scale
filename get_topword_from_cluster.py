from __future__ import absolute_import
import timeit
import argparse
from os import path
import numpy as np
import io

from autoencoder.preprocessing.preprocessing import load_corpus

from autoencoder.utils.io_utils import dump_json, load_json


def get_words(args):
    corpus = load_corpus(args.input_corpus)
    filename_corpus_dict = corpus['docs']
    vocab_dict = corpus['vocab']
    
    # we have to revort the dict
    dictionary = dict((v,k) for k, v in vocab_dict.iteritems())

    filename_label_dict = load_json(args.input_label)

    print 'Finish loading data'

    label_vocab_dict = {}

    # start counting words
    for filename in filename_corpus_dict:
        vocab_num_dict = filename_corpus_dict[filename]
        label = filename_label_dict[filename]
        try:
            label_vocab_dict[label]
        except:
            label_vocab_dict[label] = {}
        for vocab in vocab_num_dict:
            num = vocab_num_dict[vocab]
            # print 'If num is a int? : ', isinstance(num, int)
            try:
                label_vocab_dict[label][vocab] += num
            except:
                label_vocab_dict[label][vocab] = num

    print 'Finish counting word frequence'

    label_topword_dict = {}
    label_num = len(label_topword_dict)
    print 'Label num is ', label_num
    topn = args.topn
    for label in label_vocab_dict:
        vocab_num_dict = label_vocab_dict[label]
        label_topword_dict[label] = sorted(vocab_num_dict, key = vocab_num_dict.__getitem__, reverse = True)[:topn]

    print 'Finish sorting the top n word'

    dump_json(label_topword_dict, args.output_json)
    print 'Finish write the json file'

    for label in label_topword_dict:
        filename_o = args.output_dir + 'label-' + str(label) + '.txt'
        print 'filename =' , filename_o
        file_o = open(filename_o, 'w')
        for word_index in label_topword_dict[label]:
            # print 'Is word_index a int:', isinstance(word_index, int)
            text = dictionary[int(word_index)]
            text += '\n'
            file_o.write(text.encode('utf-8'))
        file_o.close()
    print 'Finish writing files!'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ic', '--input_corpus', type=str, required = True, help = 'path of the input filename corpus dict')
    parser.add_argument('-il', '--input_label', type=str, required = True, help = 'path of the input filename label dict')
    parser.add_argument('-tn', '--topn', type=int, required = True, help = 'number of top words of a cluster')
    parser.add_argument('-oj', '--output_json', type = str, required = True, help='path of the outpue json file')
    parser.add_argument('-o', '--output_dir', type=str, required = True, help = 'dir of the output top n words file')
    args = parser.parse_args()
    get_words(args)


if __name__ == '__main__':
    main()