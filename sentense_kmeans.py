from __future__ import absolute_import
import timeit
import argparse
import json
from os import path
import io
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from autoencoder.core.ae import AutoEncoder, load_ae_model, save_ae_model
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
from autoencoder.utils.op_utils import vecnorm, add_gaussian_noise, add_masking_noise, add_salt_pepper_noise
from autoencoder.utils.io_utils import dump_json


def kmeans(args):
    sentense_vec_dic = load_corpus(args.input)
    print("if sentense_vec is a dict:")
    print(isinstance(sentense_vec_dic,dict))
    vec = []
    vec_name = []
    for key in sentense_vec_dic:
        vec.append(sentense_vec_dic[key])
        vec_name.append(key)
    print "dict size is ", len(sentense_vec_dic)
    sentense_vec_X = np.array(vec)

    print "doing k-means...."
    if args.is_large_set:
        print "Do it in large data set"
        kmeans = MiniBatchKMeans(n_clusters=args.cluster_num, random_state=0).fit(sentense_vec_X)
    else:
        print "Do it in small data set"
        kmeans = KMeans(n_clusters=args.cluster_num, random_state=0).fit(sentense_vec_X)

    print "generate label"
    label_ls = kmeans.labels_

    filename_label_dic = {}
    filesize = len(sentense_vec_dic)
    for i in range(filesize):
        filename_label_dic[vec_name[i]] = int(label_ls[i])

    if args.output_json:
        print 'Write the label to the json file'
        dump_json(filename_label_dic, args.output_json)
        # with io.open(args.output_json, 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(filename_label_dic, ensure_ascii=False))
        print 'Finish writing filename_label dict to file'

    text_filename = args.text_file
    filename_text_dict = {}
    try:
        fp =  open(text_filename, 'r')
        count_doc = 0;
        while 1:
            lines = fp.readlines()
            if not lines:
                break
            for sentense in lines:
                # print(sentense)
                text = sentense.decode('utf-8').strip('\r\n')
                count_doc += 1
                doc_name = 'line-' + str(count_doc)
                filename_text_dict[doc_name] = text
    except Exception as e:
        raise e

    label_text_ls = []
    for i in range(args.cluster_num):
    	ls = []
    	label_text_ls.append(ls)

    for key in filename_label_dic:
    	label = filename_label_dic[key]
    	content = filename_text_dict[key]
    	# print 'content of ', content, 'and the label is [', label, ']'
    	label_text_ls[label].append(content)


    file_dict = {}
    for i in range(args.cluster_num):
        filename_o = args.output_dir + 'label-' + str(i) + '.txt'
        print 'filename =' , filename_o
        file_o = open(filename_o, 'w')
        for text in label_text_ls[i]:
    		text += '\n'
    		file_o.write(text.encode('utf-8'))
    	file_o.close()
    
    # for key in filename_label_dic:
    # 	label = filename_text_dict[key]
    # 	s = filename_text_dict[key]
    # 	print s
    # 	s2 = s.encode('utf-8')
    # 	file_o = file_dict[label]
    #     file_o.write(s2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required = True, help = 'path of the sentense vector file')
    parser.add_argument('-tf', '--text_file', type=str, required = True, help = 'path of the text file')
    parser.add_argument('-cn', '--cluster_num', type=int, required = True, help = 'number of cluster')
    parser.add_argument('-islarge', '--is_large_set', type = bool, default = False, help = 'Whether the data set is large')
    parser.add_argument('-o', '--output_dir', type=str, required = True, help = 'dir of the output file')
    parser.add_argument('-oj', '--output_json', type=str, help= 'path of the output json file')
    args = parser.parse_args()
    kmeans(args)


if __name__ == '__main__':
    main()