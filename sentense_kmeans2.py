from __future__ import absolute_import
import timeit
import argparse
from os import path
import numpy as np
from sklearn.cluster import KMeans

from autoencoder.core.ae import AutoEncoder, load_ae_model, save_ae_model
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
from autoencoder.utils.op_utils import vecnorm, add_gaussian_noise, add_masking_noise, add_salt_pepper_noise
from autoencoder.utils.io_utils import dump_json


def kmeans2(args):
    sentense_vec_dic = load_corpus(args.input)
    vec_name_u = load_corpus(args.question_name)
    print("if sentense_vec is a dict:")
    print(isinstance(sentense_vec_dic,dict))
    print("if vec_name is a ls:")
    print(isinstance(vec_name_u,list))
    vec = []
    vec_name = []

    for key in vec_name_u:
        filename = key.encode('utf-8')
        if filename in sentense_vec_dic.keys():
            vec.append(sentense_vec_dic[filename])
            vec_name.append(filename)

    print "file number is ", len(vec_name)
    sentense_vec_X = np.array(vec)

    print "doing k-means...."
    kmeans = KMeans(n_clusters=args.cluster_num, random_state=0).fit(sentense_vec_X)

    print "generate label"
    label_ls = kmeans.labels_

    filename_label_dic = {}
    filesize = len(vec_name)
    for i in range(filesize):
        filename_label_dic[vec_name[i]] = label_ls[i]
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
    parser.add_argument('-qn', '--question_name', type=str, required = True, help = 'path of the question name')
    parser.add_argument('-tf', '--text_file', type=str, required = True, help = 'path of the text file')
    parser.add_argument('-cn', '--cluster_num', type=int, required = True, help = 'number of cluster')
    parser.add_argument('-o', '--output_dir', type=str, required = True, help = 'dir of the output file')
    args = parser.parse_args()
    kmeans2(args)


if __name__ == '__main__':
    main()