import os
import argparse
import io
import json

from autoencoder.preprocessing.preprocessing_chinese import construct_train_corpus
from autoencoder.utils.io_utils import dump_json, load_json, write_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train_path', type=str, required=True, help='path to the training corpus')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='path to the output dir')
    parser.add_argument('-od', '--out_dict', type=str, required=True, help='dir to the output dict')
    parser.add_argument('-threshold', '--threshold', type=int, default=5, help='word frequency threshold (default 5)')
    parser.add_argument('-topn', '--topn', type=int, default=None, help='top n words (default None)')
    args = parser.parse_args()

    # train_corpus, test_corpus = construct_train_test_corpus(args.train_path, args.test_path, args.out_dir, threshold=args.threshold, topn=args.topn)
    train_corpus = construct_train_corpus(args.train_path, args.out_dir, threshold=args.threshold, topn=args.topn)
    vocab = train_corpus['vocab']
    with io.open(os.path.join(args.out_dict, 'dict.corpus'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(vocab, ensure_ascii=False))
    print 'Generate the dictionary!'

if __name__ == "__main__":
    main()
