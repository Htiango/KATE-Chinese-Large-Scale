'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adadelta
from keras.models import load_model as load_keras_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random

import numpy as np
from math import ceil

from ..utils.keras_utils import Dense_tied, KCompetitive, contractive_loss, CustomModelCheckpoint, VisualWeights


class AutoEncoder(object):
    """AutoEncoder for topic modeling.

        Parameters
        ----------
        """

    def __init__(self, input_size, dim, comp_topk=None, ctype=None, save_model='best_model'):
        self.input_size = input_size
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.save_model = save_model

        self.build()

    def build(self):
        # this is our input placeholder
        input_layer = Input(shape=(self.input_size,))

        # "encoded" is the encoded representation of the input
        if self.ctype == None:
            act = 'sigmoid'
        elif self.ctype == 'kcomp':
            act = 'tanh'
        elif self.ctype == 'ksparse':
            act = 'linear'
        else:
            raise Exception('unknown ctype: %s' % self.ctype)
        encoded_layer = Dense(self.dim, activation=act, kernel_initializer="glorot_normal", name="Encoded_Layer")
        encoded = encoded_layer(input_layer)

        if self.comp_topk:
            print 'add k-competitive layer'
            encoded = KCompetitive(self.comp_topk, self.ctype)(encoded)

        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        decoded = Dense_tied(self.input_size, activation='sigmoid', tied_to=encoded_layer, name='Decoded_Layer')(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(outputs=decoded, inputs=input_layer)

        # this model maps an input to its encoded representation
        self.encoder = Model(outputs=encoded, inputs=input_layer)

        # create a placeholder for an encoded input
        encoded_input = Input(shape=(self.dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(outputs=decoder_layer(encoded_input), inputs=encoded_input)

    def fit(self, train_X, val_X, nb_epoch=50, batch_size=100, contractive=None):
        optimizer = Adadelta(lr=2.)
        # optimizer = Adam()
        # optimizer = Adagrad()
        if contractive:
            print 'Using contractive loss, lambda: %s' % contractive
            self.autoencoder.compile(optimizer=optimizer, loss=contractive_loss(self, contractive))
        else:
            print 'Using binary crossentropy'
            self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse

        self.autoencoder.fit(train_X[0], train_X[1],
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]),
                        callbacks=[
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                                    CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto'),
                                    VisualWeights('heatmap.png', per_epoch=15)
                        ]
                        )

        return self

    def fit_generator(self, filepath_train, filepath_val, n_train, n_val, vocab_size,nb_epoch=50, batch_size=100, contractive=None):
        optimizer = Adadelta(lr=2.)
        # optimizer = Adam()
        # optimizer = Adagrad()
        if contractive:
            print 'Using contractive loss, lambda: %s' % contractive
            self.autoencoder.compile(optimizer=optimizer, loss=contractive_loss(self, contractive))
        else:
            print 'Using binary crossentropy'
            self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse

        # samples_per_epoch = ceil(n_sample / batch_size) * batch_size

        # print 'filepath =', filepath, '\n'
        print 'n_train =', n_train
        print 'n_val =', n_val
        print 'n_val / batch_size =', ceil(1.0 * n_val / batch_size)
        self.autoencoder.fit_generator(generate_arrays_from_file(filepath_train, batch_size, vocab_size), 
                        steps_per_epoch = ceil(n_train / batch_size),
                        epochs = nb_epoch,
                        validation_data=generate_arrays_from_file(filepath_val, batch_size, vocab_size),
                        validation_steps = ceil(1.0 * n_val / batch_size),
                        # workers = 10,
                        # max_q_size = 100,
                        # pickle_safe = True,
                        callbacks=[
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                                    CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')
                        ]
                        )
        return self

    def predict(self, path, vocab_size, batch_size):
        text_box = []
        for line in open(path, 'r'):
            text_box.append(line.strip())
        result = np.array([])
        first_time = True
        cnt =0
        X = []
        for i in range(len(text_box)):
            line = text_box[i]
            x, y = process_line(line, vocab_size)
            cnt += 1
            X.append(x)
            if cnt == batch_size:
                batch_doc_code = self.encoder.predict(np.r_[X])
                # print 'Write part of the doc vec'
                if first_time:
                    result = np.r_[batch_doc_code]
                    first_time = False
                else:
                    result = np.r_[np.concatenate((result, batch_doc_code), axis = 0)]
                cnt = 0
                X = []
                if (len(text_box)-i-1-batch_size < 0):
                    break;
        return result

def save_ae_model(model, model_file):
    model.save(model_file)

def load_ae_model(model_file):
    return load_keras_model(model_file, custom_objects={"KCompetitive": KCompetitive})

def process_line(line,dim):  
    arr = line.split('|')
    doc_vec = [float(val) for val in arr[0].split(' ')]
    idn_vec = [int(val) for val in arr[1].split(' ')]

    # print 'doc_vec size is ', len(doc_vec)
    # print 'index_vec size is', len(idn_vec)
    if len(doc_vec) != len(idn_vec):
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!----------------------ERROR!, array length is different'
    vec = np.zeros(dim)
    for idn, index in enumerate(idn_vec):
        # print 'index =', index
        vec[index] = doc_vec[idn]

    x = np.array(vec)  
    y = np.array(vec)  
    return x,y  
  
def generate_arrays_from_file(path,batch_size,size):  
    text_box = []
    for line in open(path,'r'):
        text_box.append(line.strip())
    while 1:
        random.shuffle(text_box)
        X =[]  
        Y =[] 
        cnt = 0 
        for i in range(len(text_box)):
            line = text_box[i]
            x, y = process_line(line, size)
            cnt += 1
            X.append(x)  
            Y.append(y) 
            if cnt==batch_size:  
                cnt = 0  
                # print 'size =',size
                yield (np.array(X), np.array(Y))  
                if (len(text_box)-i-1-batch_size<0):
                    break;
                X = []  
                Y = [] 
 
def generate_val_from_file(path,batch_size,size):  
    text_box = []
    for line in open(path,'r'):
        text_box.append(line.strip())
    while 1:
        X =[]  
        Y =[] 
        # cnt = 0 
        for i in range(len(text_box)):
            line = text_box[i]
            x, y = process_line(line, size)
            # cnt += 1
            X.append(x)  
            Y.append(y) 
        yield(np.array(X), np.array(Y))
            # if cnt==batch_size:  
            #     cnt = 0  
            #     print 'size =',size
            #     yield (np.array(X), np.array(Y))  
            #     if (len(text_box)-i-1-batch_size<0):
            #         break;
            #     X = []  
            #     Y = []            
            
        # yield (np.array(X), np.array(Y)) 
        # for i in range(len):
        #     line = text_box[i]
        #     x, y = process_line(line, size)
        #     X.append(x)  
        #     Y.append(y)
        # yield (np.array(X), np.array(Y)) 
     # while 1:  
     #    f = open(path)  
     #    cnt = 0  
     #    X =[]  
     #    Y =[]  
     #    line = f.readline()
     #    while True:
     #        x, y = process_line(line, size)  
     #        print line
     #        X.append(x)  
     #        Y.append(y)  
     #        cnt += 1  
     #        if cnt==batch_size:  
     #            cnt = 0  
     #            print 'size =',size
     #            yield (np.array(X), np.array(Y))  
     #            X = []  
     #            Y = [] 

    # while 1:  
    #     f = open(path)  
    #     cnt = 0  
    #     X =[]  
    #     Y =[]  
    #     for line in f:  
    #         # create Numpy arrays of input data  
    #         # and labels, from each line in the file 
    #         # print 'cnt =', cnt 
    #         x, y = process_line(line, size)  
    #         X.append(x)  
    #         Y.append(y)  
    #         cnt += 1  
    #         if cnt==batch_size:  
    #             cnt = 0  
    #             print 'size =',size
    #             yield (np.array(X), np.array(Y))  
    #             X = []  
    #             Y = []  
    #     f.close()  
    # while 1:
    #     f = open(path, 'r')
    #     cnt = 0  
    #     X =[]  
    #     Y =[]
    #     while 1:
    #         lines = f.readlines()
    #         if not lines:
    #             break
    #         for line in lines:
    #             # create Numpy arrays of input data  
    #             # and labels, from each line in the file  
    #             x, y = process_line(line)  
    #             X.append(x)  
    #             Y.append(y)  
    #             cnt += 1  
    #             if cnt==batch_size:  
    #                 cnt = 0  
    #                 yield (np.array(X), np.array(Y))  
    #                 X = []  
    #                 Y = []  
    # f.close()