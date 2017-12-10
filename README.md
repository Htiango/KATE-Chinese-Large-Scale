用KATE实现中文的句向量
===============

## 简介

本项目是在KATE模型的基础上进行的，在原项目的基础上实现：

+ 中文语料的句向量化
+ 用generator的方法进行模型的训练，避免因为用尽存储空间而被kill
+ 句向量结果的K-Means聚类

原始的KATE模型代码详见：[https://github.com/hugochan/KATE](https://github.com/hugochan/KATE)

KATE模型的论文详见： ["KATE: K-Competitive Autoencoder for Text"](https://arxiv.org/abs/1705.02033)



## 运行环境

This code is written in python. To use it you will need:
- Python 2.7
- A recent version of [Numpy](http://www.numpy.org)
- A recent version of [NLTK](http://www.nltk.org)
- [Tensorflow >= 1.0](https://www.tensorflow.org)
- [Keras >=2.0](https://keras.io)



## 程序运行

### 模型训练

模型训练主要分为两个步骤

+ 训练集语料生成，运行：

  ```bash
  python construct_chinese_train_corpus.py -train [train_path] -o [out_dir] -od [output_dictionary_dir] -threshold [word_freq_threshold] -topn [top_n_words]
  ```

  得到语料集以及训练集的语料表示

+ 训练KATE模型，运行：

  ```bash
  python train.py -i [train_data] -nd [num_topics] -ne [num_epochs] -bs [batch_size] -nv [num_validation] -ctype kcomp -ck [top_k] -sm [model_file] -o [output_doc_codes]
  ```

  由于在训练过程中的第一步是进行one hot，当语料库和文档集较大的时候，会出现存储空间耗尽的情况而异常退出，这里我们对这种情况采用keras下的generator的方法多次迭代进行训练，运行：

  ```bash
  python train_iterator.py -i [train_data] -tp [tmp_path(store doc_vec)] -nd [num_topics] -ne [num_epochs] -bs [batch_size] -nv [num_validation] -ctype kcomp -ck [top_k] -sm [model_file] -o [output_doc_codes]
  ```

  得到训练好的KATE模型以及训练集文档的句向量

### 模型预测

模型训练同样也分为两个步骤：

+ 测试集语料生成，需要用到之前生成训练集语料时生成的`output_dictionary`，运行：

  ```bash
  python construct_chinese_test_corpus.py -test [test_path] -vocab [corpus_path] -o [out_dir]
  ```

  得到测试集的语料表示

+ 根据之前训练得到的KATE模型得到测试集的句向量，运行：

  ```bash
  python pred.py -i [test_data] -lm [model_file] -o [output_doc_vec_file]
  ```

  同样当语料库和文档集较大的时候，会出现存储空间耗尽的情况而异常退出，这里我们采用将文档切分为几个小部分进行处理，运行：

  ```bash
  python pred_iterator.py -i [test_data] -lm [model_file] -step [sliced_dict size] -o [output_doc_vec_file]
  ```

### 文本聚类

对句向量结果，进行K-Means聚类，将对应的文档分为K份，运行：

```bash
python sentense_keans.py -i [doc_vec_file] -tf [text_file] -cn [cluster_num] -islarge [True/False] -o [out_dir] -oj [output_json]
```

得到聚类结果

之后还可以从每个cluster中提取top word，运行：

```bash
python get_topword_from_cluster.py -ic [input_corpus] -il [input_label] -tn [topn n] -oj [out_json] -o [out_dir_topn_words]
```



## Reference

Yu Chen and Mohammed J. Zaki. **"KATE: K-competitive Autoencoder for Text."** *arXiv preprint arXiv:1705.02033 (2017).*

    @article{chen2017kate,
      title={KATE: K-Competitive Autoencoder for Text},
      author={Chen, Yu and Zaki, Mohammed J},
      journal={arXiv preprint arXiv:1705.02033},
      year={2017}
    }
