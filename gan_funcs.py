import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from aux_sir_funcs import *
import random
import tensorflow.compat.v1 as tf 


def StandData(df):
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    confirmed_ = np.array(df.confirmed.fillna(0))
    fatalities_ = np.array(df.fatalities.fillna(0))

    scaler1.fit(confirmed_.reshape(-1, 1))
    confirmed_ = scaler1.transform(confirmed_.reshape(-1, 1)).reshape(-1)
    
    scaler2.fit(fatalities_.reshape(-1, 1))
    fatalities_ = scaler2.transform(fatalities_.reshape(-1, 1)).reshape(-1)
    
    df["confirmed"] = confirmed_
    df["fatalities"] = fatalities_
    idex = list(range(0, df.shape[0]))
    df["day"] = idex
    return df


class Gan_lstm(object):
    def __init__(self, sequence_length, g_len, country="US", length = 15):
        self.sequence_length = sequence_length
        self.g_len = g_len
        self.length = length
        self.country = country
        
        self.noiseSet = False #设定是 使用SIR作为噪音数据 还是  使用随机噪声
           
    def dealNoist(self, df):
        
        begin_prediction = df[df['day'] == (df.iloc[-1, -1] - self.length + 1)]['day'].values[0]
        
        rows = df[df['day'] < begin_prediction]
        confirmed_trend = [float(x) for x in rows.confirmed.values]
        mu = 0
        sigma = 0.12
        for i in range(rows.shape[0]):
            confirmed_trend.append(random.gauss(mu,sigma))
        self.noiseData = np.reshape(np.asarray(confirmed_trend), [rows.shape[0] * 2, 1])
        
#         self.noiseData = np.reshape(np.asarray(confirmed_trend), [rows.shape[0], 1])
        self.noiseSet = True #使用SIR作为噪音数据
        return self.noiseData
        
    def trainData(self, tamp_df, country, colName = "confirmed_trend"):
        self.country = country
        begin_prediction = tamp_df[tamp_df['day'] == (tamp_df.iloc[-1, -1] - self.length + 1)]['day'].values[0]
        rows = tamp_df[tamp_df['day'] < begin_prediction]
        trend_list = []
        for i in range(0, len(rows)):
            if i + self.sequence_length < len(rows):
                confirmed_trend = [float(x) for x in rows[i:i+self.sequence_length].confirmed.values]
                fatality_trend = [float(x) for x in rows[i:i+self.sequence_length].fatalities.values]

                trend_list.append({"confirmed_trend":confirmed_trend,
                                "fatality_trend":fatality_trend})

        trend_df = pd.DataFrame(trend_list)
        trend_df["temporal_inputs"] = [np.asarray([trends[colName]]) 
                                       for idx,trends in trend_df.iterrows()]
        trend_df = shuffle(trend_df)
        data = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in trend_df["temporal_inputs"].values]),
                                                              (trend_df.shape[0],1,self.sequence_length)),(0,2,1) )).astype(np.float32)
        return data
    
    def discriminator(self, input_layer, reuse=False, trainable=True):
         with tf.variable_scope("discriminator", reuse=reuse):
            
            
            
            #conv = tf.layers.conv2d(input_layer, 64, [1, 1], padding='SAME')
            conv = tf.layers.dense(input_layer, 256)
            conv = tf.nn.leaky_relu(conv)
            
            conv = tf.layers.dense(input_layer, 128)
            conv = tf.nn.leaky_relu(conv)
            
#            conv = tf.layers.conv2d(conv, 256, [2, 1], padding='SAME')
#             conv = tf.layers.batch_normalization(conv, trainable=trainable)
#            conv = tf.nn.leaky_relu(conv)
            
#            conv = tf.layers.conv2d(conv, 512, [2, 1], padding='SAME')
#             conv = tf.layers.batch_normalization(conv, trainable=trainable)
#            conv = tf.nn.relu(conv)
            conv = tf.layers.dense(input_layer, 64)
            conv = tf.nn.leaky_relu(conv)
        
            conv = tf.layers.flatten(conv)
            conv = tf.layers.dense(conv, 1)
            
        
            return conv
    
    def generator(self, input_layer, reuse=False, trainable=True):
        
        with tf.variable_scope("generator", reuse=reuse):
            
            x_o = tf.layers.dense(input_layer, 
                                    self.sequence_length * 1024 * 1)
            
            x_o  = tf.reshape(x_o, [-1, self.sequence_length, 1024])
#             flat = tf.layers.batch_normalization(flat, trainable=trainable)
            #flat = tf.nn.relu(flat)

            # create 2 LSTMCells
            rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 64]]
    
            # create a RNN cell composed sequentially of a number of RNNCells
            multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)
            # 'outputs' is a tensor of shape [batch_size, max_time, 256]
            # 'state' is a N-tuple where N is the number of LSTMCells containing a
            # tf.nn.rnn_cell.LSTMStateTuple for each cell
            x_o, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                                         inputs=x_o,
                                                         dtype=tf.float32)

            print('after lstm', x_o.shape)
            '''
            tconv = tf.layers.conv2d_transpose(flat, 512, 
                                               kernel_size=[1, 1], 
                                               strides=[1, 1], padding='SAME')
#             tconv = tf.layers.batch_normalization(tconv, trainable=trainable)
            tconv = tf.nn.relu(tconv)
    
            
            tconv = tf.layers.conv2d_transpose(tconv, 256, 
                                               kernel_size=[1, 1], 
                                               strides=[1, 1], padding='SAME')
#             tconv = tf.layers.batch_normalization(tconv, trainable=trainable)
            tconv = tf.nn.relu(tconv)
    
            
            tconv = tf.layers.conv2d_transpose(tconv, 128, 
                                               kernel_size=[1, 1], 
                                               strides=[1, 1], padding='SAME')
#             tconv = tf.layers.batch_normalization(tconv, trainable=trainable)
            tconv = tf.nn.relu(tconv)
    
            tconv = tf.layers.dense(tconv, 1)
            
            tconv = tf.layers.conv2d_transpose(tconv, 1, 
                                               kernel_size=[1, 1], 
                                               strides=[1, 1], padding='SAME')
#             tconv = tf.layers.batch_normalization(tconv, trainable=trainable)
            '''
            x_o = tf.layers.dense(x_o, 1)
            x_o = tf.nn.tanh(x_o, name="gen_data")
            print('fin', x_o.shape)
        return x_o
    
    def plot_loss(self):
        fig = plt.figure(figsize=[20, 5])

        plt.subplot(121)
        plt.plot(self.all_ds)
        plt.title('Loss over epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['d_loss'], loc='best')

        plt.subplot(122)
        plt.plot(self.all_gs)
        plt.title('Loss over epochs for the number of cases')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['g_loss', 'Validation'], loc='best')

        plt.show()
        
    def train(self, data, iter_epoch=20, 
              learning_rate=0.00002, beta1=0.5):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, self.sequence_length, 1, 1], name="Data")
        noise = tf.placeholder(tf.float32, shape=[None, self.g_len], name="Noise")
        
        G = self.generator(noise, reuse=False)
        print(G.shape)
        D_real = self.discriminator(x, reuse=False)
        D_fake = self.discriminator(G, reuse=True)

        D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        D_loss = D_real_loss + D_fake_loss

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta1=beta1).minimize(G_loss, var_list=g_vars)
            d_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta1=beta1).minimize(D_loss, var_list=d_vars)
        print("network build success")
        
        self.all_ds = []
        self.all_gs = []
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iter_epoch):
                total_batch = int(data.shape[0])
                
                nsl = self.random_noise(total_batch)
                ds, _ = sess.run([D_loss, d_op], feed_dict={x: data, noise: nsl})
                
                for j in range(3):
                    ns2 = np.random.uniform(low=-1, high=1, size=[total_batch, self.g_len])
                    gs, _ = sess.run([G_loss, g_op], feed_dict={x: data, noise: ns2})
                
                self.all_ds.append(ds)
                self.all_gs.append(gs)
                print("Iter: {}      d_loss: {:.4}       g_loss: {:.4}".format(i, ds, gs))
                
            saver.save(sess, "./gan_model/" + self.country + "_model")
            
    def random_noise(self, num):
        if self.noiseSet == True:
            bk  = [random.randint(0,self.noiseData.shape[0] - 1) for _ in range(num * self.g_len)]
            ns1 = np.reshape(self.noiseData[bk, :], [num, self.g_len])
        else:
            ns1 = np.random.uniform(low=-1, high=1, size=[num, self.g_len])
        return ns1
            
    def predict(self, num):
        
        ns1 = self.random_noise(num)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "./gan_model/" + self.country + "_model")

            graph = tf.get_default_graph()
            noise = graph.get_tensor_by_name("Noise:0")
            data_ = graph.get_tensor_by_name("generator/gen_data:0")
            feed_dict ={noise : ns1}
            data = sess.run(data_,feed_dict)
        return np.reshape(data, [data.shape[0], data.shape[1], data.shape[2]])


    
class Gan(object):
    def __init__(self, sequence_length, g_len, country="US", length = 15):
        self.sequence_length = sequence_length
        self.g_len = g_len
        self.length = length
        self.country = country
        
        self.noiseSet = False #设定是 使用SIR作为噪音数据 还是  使用随机噪声
           
    def dealNoist(self, df):
        
        begin_prediction = df[df['day'] == (df.iloc[-1, -1] - self.length + 1)]['day'].values[0]
        
        rows = df[df['day'] < begin_prediction]
        confirmed_trend = [float(x) for x in rows.confirmed.values]
        mu = 0
        sigma = 0.12
        for i in range(rows.shape[0]):
            confirmed_trend.append(random.gauss(mu,sigma))
        self.noiseData = np.reshape(np.asarray(confirmed_trend), [rows.shape[0] * 2, 1])
        
#         self.noiseData = np.reshape(np.asarray(confirmed_trend), [rows.shape[0], 1])
        self.noiseSet = True #使用SIR作为噪音数据
        return self.noiseData
        
    def trainData(self, tamp_df, country, colName = "confirmed_trend"):
        self.country = country
        begin_prediction = tamp_df[tamp_df['day'] == (tamp_df.iloc[-1, -1] - self.length + 1)]['day'].values[0]
        rows = tamp_df[tamp_df['day'] < begin_prediction]
        trend_list = []
        for i in range(0, len(rows)):
            if i + self.sequence_length < len(rows):
                confirmed_trend = [float(x) for x in rows[i:i+self.sequence_length].confirmed.values]
                fatality_trend = [float(x) for x in rows[i:i+self.sequence_length].fatalities.values]

                trend_list.append({"confirmed_trend":confirmed_trend,
                                "fatality_trend":fatality_trend})

        trend_df = pd.DataFrame(trend_list)
        trend_df["temporal_inputs"] = [np.asarray([trends[colName]]) 
                                       for idx,trends in trend_df.iterrows()]
        trend_df = shuffle(trend_df)
        data = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in trend_df["temporal_inputs"].values]),
                                                              (trend_df.shape[0],1,self.sequence_length)),(0,2,1) )).astype(np.float32)
        return data
    
    def discriminator(self, input_layer, reuse=False, trainable=True):
         with tf.variable_scope("discriminator", reuse=reuse):
        
            conv = tf.layers.conv2d(input_layer, 64, [1, 1], padding='SAME')
#             conv = tf.layers.batch_normalization(conv, trainable=trainable)
            conv = tf.nn.relu(conv)
            
            conv = tf.layers.conv2d(conv, 128, [1, 1], padding='SAME')
#             conv = tf.layers.batch_normalization(conv, trainable=trainable)
            conv = tf.nn.relu(conv)
            
            conv = tf.layers.conv2d(conv, 256, [2, 1], padding='SAME')
#             conv = tf.layers.batch_normalization(conv, trainable=trainable)
            conv = tf.nn.relu(conv)
            
            conv = tf.layers.conv2d(conv, 512, [2, 1], padding='SAME')
#             conv = tf.layers.batch_normalization(conv, trainable=trainable)
            conv = tf.nn.relu(conv)
            
            flat = tf.layers.flatten(conv)
            dense = tf.layers.dense(flat, 1)
            
        
            return dense
    
    def generator(self, input_layer, reuse=False, trainable=True):
        with tf.variable_scope("generator", reuse=reuse):
            
            dense = tf.layers.dense(input_layer, self.sequence_length * 1024 * 1)
            
            flat  = tf.reshape(dense, [-1, self.sequence_length, 1, 1024])
#             flat = tf.layers.batch_normalization(flat, trainable=trainable)
            flat = tf.nn.relu(flat)
            
            tconv = tf.layers.conv2d_transpose(flat, 512, kernel_size=[1, 1], strides=[1, 1], padding='SAME')
#             tconv = tf.layers.batch_normalization(tconv, trainable=trainable)
            tconv = tf.nn.relu(tconv)
            
            tconv = tf.layers.conv2d_transpose(tconv, 256, kernel_size=[1, 1], strides=[1, 1], padding='SAME')
#             tconv = tf.layers.batch_normalization(tconv, trainable=trainable)
            tconv = tf.nn.relu(tconv)
            
            tconv = tf.layers.conv2d_transpose(tconv, 128, kernel_size=[1, 1], strides=[1, 1], padding='SAME')
#             tconv = tf.layers.batch_normalization(tconv, trainable=trainable)
            tconv = tf.nn.relu(tconv)
    
            tconv = tf.layers.conv2d_transpose(tconv, 1, kernel_size=[1, 1], strides=[1, 1], padding='SAME')
#             tconv = tf.layers.batch_normalization(tconv, trainable=trainable)
            tconv = tf.nn.relu(tconv, name="gen_data")
    
        return tconv
    
    def plot_loss(self):
        fig = plt.figure(figsize=[20, 5])

        plt.subplot(121)
        plt.plot(self.all_ds)
        plt.title('Loss over epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['d_loss'], loc='best')

        plt.subplot(122)
        plt.plot(self.all_gs)
        plt.title('Loss over epochs for the number of cases')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['g_loss', 'Validation'], loc='best')

        plt.show()
        
    def train(self, data, iter_epoch=20, 
              learning_rate=0.00002, beta1=0.5):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, self.sequence_length, 1, 1], name="Data")
        noise = tf.placeholder(tf.float32, shape=[None, self.g_len], name="Noise")
        
        G = self.generator(noise, reuse=False)
        print(G.shape)
        D_real = self.discriminator(x, reuse=False)
        D_fake = self.discriminator(G, reuse=True)

        D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        D_loss = D_real_loss + D_fake_loss

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta1=beta1).minimize(G_loss, var_list=g_vars)
            d_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta1=beta1).minimize(D_loss, var_list=d_vars)
        print("network build success")
        
        self.all_ds = []
        self.all_gs = []
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iter_epoch):
                total_batch = int(data.shape[0])
                
                nsl = self.random_noise(total_batch)
                ds, _ = sess.run([D_loss, d_op], feed_dict={x: data, noise: nsl})
                
                for j in range(3):
                    ns2 = np.random.uniform(low=-1, high=1, size=[total_batch, self.g_len])
                    gs, _ = sess.run([G_loss, g_op], feed_dict={x: data, noise: ns2})
                
                self.all_ds.append(ds)
                self.all_gs.append(gs)
                print("Iter: {}      d_loss: {:.4}       g_loss: {:.4}".format(i, ds, gs))
                
            saver.save(sess, "./gan_model/" + self.country + "_model")
            
    def random_noise(self, num):
        if self.noiseSet == True:
            bk  = [random.randint(0,self.noiseData.shape[0] - 1) for _ in range(num * self.g_len)]
            ns1 = np.reshape(self.noiseData[bk, :], [num, self.g_len])
        else:
            ns1 = np.random.uniform(low=-1, high=1, size=[num, self.g_len])
        return ns1
            
    def predict(self, num):
        
        ns1 = self.random_noise(num)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "./gan_model/" + self.country + "_model")

            graph = tf.get_default_graph()
            noise = graph.get_tensor_by_name("Noise:0")
            data_ = graph.get_tensor_by_name("generator/gen_data:0")
            feed_dict ={noise : ns1}
            data = sess.run(data_,feed_dict)
        return np.reshape(data, [data.shape[0], data.shape[1], data.shape[2]])
    
    
    
def ganData(seq, lens, df, name, noiseSet = True, 
            colName = "confirmed_trend", iter_epoch=10,
            gan_t = 'lstm'):
    tf.compat.v1.disable_eager_execution()
    
    temp_data = df.copy()
    temp_data = temp_data.iloc[:-30]
    print(temp_data.shape)
    print(gan_t)
    if gan_t=='lstm':
        gan = Gan_lstm(seq, lens)
    else:
        gan = Gan(seq, lens)
        
    temp_data = StandData(temp_data)
#     temp_data = log2_(df)
    
    data = gan.trainData(temp_data, name, colName)
    data = np.reshape(data[:, :, :], [data.shape[0], data.shape[1], data.shape[2], 1])
    N = test(name, df)
    
    if noiseSet == True:
        sirData = deal_SIR(temp_data.copy(), name, N=N)
        noise = gan.dealNoist(sirData)
    
    gan.train(data, iter_epoch=iter_epoch)
    gan.plot_loss()
    return gan