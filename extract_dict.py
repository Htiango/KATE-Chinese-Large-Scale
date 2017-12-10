import io
import json
import os
import re
import string
import argparse
from autoencoder.preprocessing.preprocessing import load_corpus
from autoencoder.utils.io_utils import dump_json, load_json, write_file


def extract_dict(args):
    corpus = load_corpus(args.input)
    vocab = corpus['vocab']
    with io.open(os.path.join(args.output_dir, 'dict.corpus'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(vocab, ensure_ascii=False))
    print 'Generate the dictionary!'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required = True, help = 'path of the corpus containing the dict')
    parser.add_argument('-o', '--output_dir', type=str, required = True, help = 'dir of the output dict file')
    args = parser.parse_args()
    extract_dict(args)

if __name__ == '__main__':
    main()