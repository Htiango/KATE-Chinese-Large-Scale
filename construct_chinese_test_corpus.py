import os
import argparse

from autoencoder.preprocessing.preprocessing_chinese import construct_test_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', '--test_path', type=str, required=True, help='path to the training corpus')
    parser.add_argument('-vocab', '--vocab_path', type=str, required=True, help='path to the dict corpus')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='path to the output dir')
    args = parser.parse_args()

    print 'Test path = ', args.test_path
    print 'Dict path = ', args.vocab_path
    print 'Output path = ', args.out_dir
    test_corpus = construct_test_corpus(args.test_path, args.vocab_path, args.out_dir)

if __name__ == "__main__":
    main()
