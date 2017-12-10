'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import timeit
import argparse
import os
import numpy as np
import matplotlib
import random
from math import ceil
matplotlib.use('Agg')

from autoencoder.core.ae import AutoEncoder, load_ae_model, save_ae_model
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
from autoencoder.utils.op_utils import vecnorm, add_gaussian_noise, add_masking_noise, add_salt_pepper_noise
from autoencoder.utils.io_utils import dump_json

def doc2vec_dict(doc):
    dic = {}
    for idx, val in doc.items():
        dic[int(idx)] = val
    return dic

def train_iterator(args):
    corpus = load_corpus(args.input)
    print 'Successfully Load the vocab and the docs!'
    n_vocab, docs_origin = len(corpus['vocab']), corpus['docs']
    corpus.clear() # save memory

    docs = {}
    for key in docs_origin:
        if len(docs_origin[key]) != 0:            
            docs[key] = docs_origin[key]
        # else:
        #     print key
    print 'Delete the empty docs'
    doc_keys = docs.keys()
    # random.shuffle(doc_keys) # random and shuffle the data
    print 'Shuffle the data!'
    n_samples = len(doc_keys)
    print 'The sample number is', n_samples
    np.random.seed(0)
    val_idx = set(np.random.choice(range(n_samples), args.n_val, replace=True))
    train_idx = set(range(n_samples)) - set(val_idx)
    print 'Generate random val index as well as train index!'

    temp_path_train = os.path.join(args.temp_path, 'temp_train.dat')
    temp_path_val = os.path.join(args.temp_path, 'temp_val.dat')

    try:
        file_temp_train = open(temp_path_train, 'w')
        file_temp_val = open(temp_path_val, 'w')

        for idx,k in enumerate(doc_keys):
            if idx % 100000 == 0:
                print 'Doc', idx

            doc_dict = doc2vec_dict(docs[k])
            doc_dict_keys = doc_dict.keys()
            doc_v = []
            for key in doc_dict_keys:
                doc_v.append(doc_dict[key])
            doc_v = np.r_[doc_v]

            # -------------------test here-----------
            # print 'doc vec len = ', len(doc_v)
            # print 'doc_key =', k
            # print docs[k] 
            if len(doc_v) == 0:
                print 'doc_key =', k
                continue
            # ----------------------------------

            X_doc = vecnorm(doc_v, 'logmax1', 0)
            del docs[k]

            s_doc = ' '.join(str(i) for i in X_doc)
            s_idn = ' '.join(str(i) for i in doc_dict_keys)
            s = s_doc + '|' + s_idn + '\n'

            if idx in train_idx:
                file_temp_train.write(s)
            else:
                file_temp_val.write(s)
        file_temp_val.close()
        file_temp_train.close()
        print 'Finish writing to temp path!'
    except Exception, e:
        raise

    start = timeit.default_timer()
    ae = AutoEncoder(n_vocab, args.n_dim, comp_topk=args.comp_topk, ctype=args.ctype, save_model=args.save_model)
    ae.fit_generator(temp_path_train, temp_path_val, len(train_idx), len(val_idx),n_vocab,nb_epoch=args.n_epoch, batch_size=args.batch_size, contractive=args.contractive)

    print 'runtime: %ss' % (timeit.default_timer() - start)

    if args.output:
        train_doc_codes = ae.predict(temp_path_train, n_vocab, args.batch_size)
        print 'Generate the train doc vec'
        val_doc_codes = ae.predict(temp_path_val, n_vocab, args.batch_size)
        print 'Generate the val doc vec'

        # train_doc_codes = ae.encoder.predict(X_train)
        # val_doc_codes = ae.encoder.predict(X_val)
        doc_keys = np.array(doc_keys)
        dump_json(dict(zip(doc_keys[list(train_idx)].tolist(), train_doc_codes.tolist())), args.output + 'train_doc_vec')
        dump_json(dict(zip(doc_keys[list(val_idx)].tolist(), val_doc_codes.tolist())), args.output + 'val_doc_vec')
        print 'Saved doc codes file to %s and %s' % (args.output + 'train_doc_vec', args.output + 'val_doc_vec')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus file')
    parser.add_argument('-tp', '--temp_path', type=str, required=True, help='temp_path storing the doc vec')
    parser.add_argument('-nd', '--n_dim', type=int, default=128, help='num of dimensions (default 128)')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='size of validation set (default 1000)')
    parser.add_argument('-ck', '--comp_topk', type=int, help='competitive topk')
    parser.add_argument('-ctype', '--ctype', type=str, help='competitive type (kcomp, ksparse)')
    parser.add_argument('-sm', '--save_model', type=str, default='model', help='path to the output model')
    parser.add_argument('-contr', '--contractive', type=float, help='contractive lambda')
    parser.add_argument('--noise', type=str, help='noise type: gs for Gaussian noise, sp for salt-and-pepper or mn for masking noise')
    parser.add_argument('-o', '--output', type=str, help='path to the output doc codes file')
    args = parser.parse_args()

    if args.noise and not args.noise in ['gs', 'sp', 'mn']:
        raise Exception('noise arg should left None or be one of gs, sp or mn')
    train_iterator(args)

if __name__ == '__main__':
    main()
