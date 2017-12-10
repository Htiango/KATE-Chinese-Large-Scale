import io
import json
import os
import re
import string
import argparse
import random

def get_all_files(corpus_path, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(corpus_path) for file in filenames if os.path.isfile(os.path.join(root, file)) and not file.startswith('.')]
    else:
        return [os.path.join(corpus_path, filename) for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)) and not filename.startswith('.')]


def select_question(args):
    files = get_all_files(args.input, False)
    output_pre = args.output_dir
    num = args.selected_num
    for filename in files:
        try:
            with open(filename, 'r') as fp:
                lines = fp.readlines()
                if len(lines) < num:
                    question_ls = lines
                else:
                    question_ls = random.sample(lines, args.selected_num)
                parent_name, child_name = os.path.split(filename)
                outputname = output_pre + child_name

                file_o = open(outputname, 'w')
                for text in question_ls:
                    text += '\n'
                    file_o.write(text)
                    # file_o.write(text.encode('utf-8'))

                file_o.close()
                print 'finish selecting:' , outputname
        except Exception as e:
            raise e
    print 'Finish the selecting!'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required = True, help = 'dir of the files')
    parser.add_argument('-n', '--selected_num', type=int, required = True, help = 'number of the selected question')
    parser.add_argument('-o', '--output_dir', type=str, required = True, help = 'dir of the output dict file')
    args = parser.parse_args()
    select_question(args)

if __name__ == '__main__':
    main()