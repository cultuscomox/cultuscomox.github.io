import numpy as np
import scipy as sp
import tensorflow as tf
class LinearModel:
    def __init__(self, loss_fun=tf.losses.mean_squared_error,sess=None):
        if sess==None:
            sess=tf.Session()
        self.loss_fun = loss_fun
        self.sess = sess
        
    def fit(self, data, targets, n_epochs=10, batch_size=200, learning_rate=1e-3, init_stddev=1e-1, use_dropout=True, dropout_stddev=1e-2, l2_penalty=1e-1):
        n,xdim = data.shape
        ny,ydim = targets.shape
        self.ydim=ydim
        n_batches = int(np.ceil((n*n_epochs)/float(batch_size)))
        self.w = tf.Variable(np.random.normal(scale=init_stddev,
        size=[xdim,ydim]).astype('float32'), name='w')
        self.b = tf.Variable(np.random.normal(scale=init_stddev,size=[ydim]).astype('float32'), name='b')
        self.X = tf.placeholder(tf.float32, shape=[None,xdim], name='X')
        self.Y_target = tf.placeholder(tf.float32, shape=[None,ydim], name='Y')
        #w is xdim, ydi
        self.score_no_bias = tf.matmul(self.X,self.w)
        self.score = self.score_no_bias + self.b
        self.score_drop = self.score_no_bias*tf.random_normal(shape=tf.shape(self.score_no_bias),stddev=dropout_stddev,mean=1.0) + self.b
        self.loss = tf.reduce_mean(self.loss_fun(self.Y_target,self.score)) + l2_penalty*tf.reduce_sum(tf.square(self.w))
        self.loss_drop = tf.reduce_mean(self.loss_fun(self.Y_target,self.score_drop)) + l2_penalty*tf.reduce_sum(tf.square(self.w))
#        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)
        if (use_dropout==True):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_drop)
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        i = 0
        for _ in xrange(n_batches):
            ii = (i + batch_size ) % n
            self.fit_batch(data[i:ii,:], targets[i:ii,:])
            i = ii
#            print self.sess.run(self.loss,feed_dict={self.X:data,self.Y_target:targets})
        return self
    
    def fit_batch(self, data_batch, target_batch):
        feed_dict = {self.X:data_batch,self.Y_target:target_batch}
        self.sess.run(self.optimizer, feed_dict=feed_dict)

    def predict_samples(self, data, n_samps=100):
        (dx,dy) = data.shape
        pred_samples = np.zeros([n_samps,dx,self.ydim],dtype=float)
        for i in xrange(0,n_samps):
            pred_samples[i,:,:] = self.sess.run(self.score_drop,feed_dict={self.X:data})
        return pred_samples

        
    def predict(self, data):
            return self.sess.run(self.score, feed_dict={self.X:data})
        
def logistic_loss(labels,score):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=score)

       

class LinearVariationalModel(LinearModel):
    def __init__(self, loss_fun, xdim, ydim, drop_std=0.0, init_stddev=1e-5):
        LinearModel.__init__(self,loss_fun,xdim,ydim,init_stddev)
        #one random sample per minibatch = less variance
        if adapt_dropout:
            self.drop_std = tf.Variable(drop_std)
        else:
            self.drop_std = drop_std
        self.score_drop_per_datum = (1. + tf.random_normal(tf.shape(self.score_no_bias)))*self.drop_std * self.score_no_bias + self.b
        self.loss_drop = tf.reduce_mean(loss_fun(self.Y_target,self.score_drop_per_datum))
#        self.score_drop_per_batch = 

#tensorflow insists that the arguments are named with these particular functions
#for some reason
def ce_loss(labels, score):
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=score)

def mhinge_loss(labels, score):
    max_arg = score + 1 - labels
    return tf.reduce_max(max_arg, axis=1) - tf.reduce_sum(score*labels, axis=1)
def mse(target, score):
    return tf.reduce_mean((target-score)**2)
