#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @auther: RuisongZhou
# @date: 2/8/2019


import tensorflow as tf
import numpy as np
import pdb

class TextCNN():
    __shuffer_falg = False
    __static_falg = True

    def __init__(self, W_list, shuffer_falg, static_falg, filter_numbers, filter_sizes, sentence_length, \
        embedding_size=300, learning_rate=0.05, epochs=10, batch_size=50, dropout_pro=0.5):
        '''
        W_list: embedding layer's param
        shuffer_flag: shuffer the data or not
        static_flag: use static train or not

        '''
        self.__shuffer_falg = shuffer_falg
        self.__static_falg = static_falg
        self.learning_rate_item = learning_rate
        self.epochs = epochs
        self.sentence_length = sentence_length
        self.filter_numbers = filter_numbers
        self.batch_size = batch_size
        self.dropout_pro_item = dropout_pro
        self.embedding_size = embedding_size

        # 1. setting graph
        tf.reset_default_graph()
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # 1. input layer
            self.input_x = tf.placeholder(dtype=tf.int32,shape=[None,sentence_length],name='input_x')
            self.input_y = tf.placeholder(dtype=tf.int32,shape=[None, 2],name='input_y')
            self.dropout_pro = tf.placeholder(dtype=tf.float32,name='dropout_pro')
            self.l2_loss = tf.constant(0.0)
            


            # 2. embedding_layer
            with tf.name_scope('embedding_layer'):
                train_bool = not self.__static_falg

                self.embedding_layer_W = tf.Variable(initial_value=W_list, dtype=tf.float32, trainable=train_bool, name='embedding_layer_W')
                self.embedding_layer_layer = tf.nn.embedding_lookup(self.embedding_layer_W,self.input_x)
                self.embedding_layer_expand = tf.expand_dims(self.embedding_layer_layer, -1)


            #3 conv layer + maxpool layer for each filer size
            with tf.name_scope('dropout_layer'):
                pool_layer_list = []
                for filter_size in filter_sizes:
                    max_pool_layer = self.add_conv_layer(filter_size, filter_numbers)
                    pool_layer_list.append(max_pool_layer)
            
            # 4. Fully-connection layer
            # use softmax and L2 normolization
            # combine all the max pool --feature




        
        def train(self,train_x,train_y):
            pass
        

        def validatation(self,text_x, text_y):
            pass

        def close(self):
            pass

        def get_batches(self):
            pass
        
        def add_conv_layer(self):
            pass

        def variables_summery(self):
            pass