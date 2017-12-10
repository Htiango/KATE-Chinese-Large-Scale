import io
import json
import os
import re
import string
import argparse
import jieba

from autoencoder.utils.io_utils import dump_json, load_json, write_file

def load_question_words(file):
    question_words = []
    try:
        with open(file, 'r') as f:
            for line in f:
                # print(line);
                question_words.append(line.strip('\n ').decode('utf-8'))
    except Exception as e:
        raise e
    question_words_set = set()
    for word in question_words:
        question_words_set.add(word)
    return question_words_set

def init_question_words(path):
    try:
        cached_question_words = load_question_words(os.path.join(os.path.split(__file__)[0], path))
        print 'Loaded question words'
    except:
        print 'No question words!'

    return cached_question_words

def contain_question_words(text, question_words):
    words = []
    try:
        s_list = jieba.lcut(text, cut_all = False)
        for token in s_list:
            if token in question_words :
                return True
        # print('Finish tokenize!')
        return False
    except Exception as e:
        raise e

def extract_question(args):
    inputpath = args.input
    questionpath = args.input_question
    outputpath  =args.output_text
    question_name = []
    try:
        f_in =  open(inputpath, 'r')
        f_out = open(outputpath, 'w')
        count_doc = 0;
        question_words = init_question_words(questionpath)
        while 1:
            lines = f_in.readlines()
            if not lines:
                break
            for sentense in lines:
                # print(sentense)
                text = sentense.decode('utf-8').strip('\r\n')

                count_doc += 1
                doc_name = 'line-' + str(count_doc)

                if contain_question_words(text, question_words):
                	question_name.append(doc_name.decode('utf-8'))
                	text += '\n'
                	f_out.write(text.encode('utf-8'))

                if count_doc % 10000 == 0:
                    print doc_name
        f_out.close()
        f_in.close()
        print 'Finish writing the question text file'
        with io.open(os.path.join(args.output_json, 'question_name.corpus'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(question_name, ensure_ascii=False))
        print 'Finish writing the question name json file'
    except Exception as e:
        raise e
    
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required = True, help = 'path of the original input text file')
    parser.add_argument('-iq', '--input_question', type=str, required = True, help = 'path of the input question')
    parser.add_argument('-oj', '--output_json', type=str, required = True, help = 'the output dir of the file which contains the name')
    parser.add_argument('-ot', '--output_text', type=str, required = True, help = 'the output path of the selected quesetion ')
    args = parser.parse_args()
    extract_question(args)

if __name__ == '__main__':
    main()