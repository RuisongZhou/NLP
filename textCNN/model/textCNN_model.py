#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @auther: RuisongZhou
# @date: 2/9/2019


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
            self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
            self.l2_loss = tf.constant(0.0)
            self.embedding_layer = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, sentence_length, embedding_size],
                 name='embedding_layer')


            # 2. embedding_layer
            with tf.name_scope('embedding_layer'):
                train_bool = not self.__static_falg

                self.embedding_layer_W = tf.Variable(initial_value=W_list, dtype=tf.float32, trainable=train_bool, name='embedding_layer_W')
                self.embedding_layer_layer = tf.nn.embedding_lookup(self.embedding_layer_W,self.input_x)
                self.embedding_layer_expand = tf.expand_dims(self.embedding_layer_layer, -1)


            #3 conv layer + maxpool layer for each filer size
                pool_layer_list = []
                for filter_size in filter_sizes:
                    max_pool_layer = self.add_conv_layer(filter_size, filter_numbers)
                    pool_layer_list.append(max_pool_layer)
            

            # 4. Fully-connection layer
            # use softmax and L2 normolization
            # combine all the max pool --feature
            with tf.name_scope('dropout_layer'):
                
                max_num = len(filter_sizes)*self.filter_numbers
                h_pool = tf.concat(pool_layer_list,name='last_pool_layer', axis = 3)
                pool_layer_flat = tf.reshape(h_pool,[-1,max_num],name='pool_layer_flat')

                dropout_pro_layer = tf.nn.dropout(pool_layer_flat,self.dropout_pro,name='droupout')
            
            with tf.name_scope('soft_max_layer'):
                softmax_w = tf.Variable(tf.truncated_normal([max_num,2],stddev=0.01),name='softmax_liner_weight')
                self.variables_summery(softmax_w)

                softmax_b = tf.Variable(tf.constant(0.1,shape=[2]), name='softmax_linear_bias')
                self.variables_summery(softmax_b)

                self.l2_loss += tf.nn.l2_loss(softmax_w)
                self.l2_loss += tf.nn.l2_loss(softmax_b)

                self.softmax_values = tf.nn.xw_plus_b(dropout_pro_layer,softmax_w,softmax_b,name='soft_values')
                self.predictions = tf.argmax(self.softmax_values, axis=1, name='predictions', output_type=tf.int32)


            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_values,labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + 0.001*self.l2_loss

                tf.summary.scalar('last_loss', self.loss  )
            
            with tf.name_scope('accuracy'):
                correct_acc = tf.equal(self.predictions,tf.argmax(self.input_y,axis=1,output_type=tf.int32))

                self.accuracy = tf.reduce_mean(tf.cast(correct_acc,'float'),name='accuracy')
                tf.summary.scalar('accuracy',self.accuracy)

            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_op = optimizer.minimize(self.loss)



            self.session = tf.InteractiveSession(graph=self.train_graph)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('../result/', graph=self.train_graph)



    
    def train(self,train_x,train_y):
        self.session.run(tf.global_variables_initializer())
        #迭代训练
        for epoch in range(self.epochs):
            # pdb.set_trace()
            train_batch = self.get_batches(train_x, train_y, self.batch_size)
            train_loss, train_acc, count = 0.0, 0.0, 0
            for batch_i in range(len(train_x)//self.batch_size):
                x,y = next(train_batch)
                feed = {
                    self.input_x:x,
                    self.input_y:y,
                    self.dropout_pro:self.dropout_pro_item,
                    self.learning_rate:self.learning_rate_item
                }
                _,summarys,loss,accuracy = self.session.run([self.train_op,self.merged,self.loss,self.accuracy],feed_dict=feed)
                train_loss, train_acc, count = train_loss + loss, train_acc + accuracy, count + 1
                self.train_writer.add_summary(summarys,epoch)
                # each 5 batch print log
                if (batch_i+1) % 15 == 0:
                    print('Epoch {:>3} Batch {:>4}/{} train_loss = {:.3f} accuracy = {:.3f}'.
                          format(epoch,batch_i,(len(train_x)//self.batch_size),train_loss/float(count),train_acc/float(count)))

    
    def validation(self,text_x, text_y):
        test_batch = self.get_batches(text_x, text_y, self.batch_size)
        eval_loss, eval_acc, count = 0.0, 0.0, 0.0
        for batch in range(len(text_x)// self.batch_size):
            x,y = next(test_batch)
            feed = {
                self.embedding_layer: x,
                self.input_y :y,
                self.dropout_pro: self.dropout_pro_item,
                self.learning_rate:1.0
            }
            loss, accuracy = self.session.run([self.loss, self.accuracy],feed_dict=feed)
            eval_loss, eval_acc, count = eval_loss+loss, eval_acc+accuracy, count+1

        return eval_acc/float(count), eval_loss/float(count)
            

    def close(self):
        self.session.close()
        self.train_writer.close()

    #generate batches
    def get_batches(self,Xs, Ys, batch_size):
        for start in range(0, len(Xs), batch_size):
            end = min(start+batch_size,len(Xs))
            yield Xs[start:end], Ys[start:end]
    
    def add_conv_layer(self,filter_size,filter_numbers):
        with tf.name_scope('conv_maxpool-size%d'%(filter_size)):
            #卷积层
            filter_shape = [filter_size, self.embedding_size, 1, filter_numbers]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter_weight')
            self.variables_summery(W)
            b = tf.Variable(tf.constant(0.1, shape=[filter_numbers], name='filter_bias'))
            self.variables_summery(b)


            #参数说明
            #第一个参数input：指需要做卷积的输入图像 [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
            #第二个参数filter：相当于CNN中的卷积核 [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
            #第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4,
            #第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
            #第五个参数：use_cudnn_on_gpu: bool类型，是否使用cudnn加速，默认为true
            conv_layer = tf.nn.conv2d(self.embedding_layer_expand,W,strides=[1,1,1,1],padding='VALID',name='conv_layer')
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer,b),name='relu_layer')

            max_pool_layer = tf.nn.max_pool(relu_layer,ksize=[1, self.sentence_length-filter_size+1, 1, 1],strides=[1,1,1,1],padding='VALID',name='maxpool')
            return max_pool_layer


    def variables_summery(self,var):
        """
        :param var: Tensor, Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summeries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)  # 记录参数的均值

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))

                # 用直方图记录参数的分布
                tf.summary.histogram('histogram', var)