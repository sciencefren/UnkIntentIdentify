import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
    
    
class DEC(object):
    def __init__(self, params):
        self.n_cluster = params["n_clusters"]
        self.kmeans = KMeans(n_clusters=params["n_clusters"], n_init=20)
        self.alpha = params['alpha']
        
        self.mu = tf.Variable(tf.zeros(shape=(params["n_clusters"], params["feature_dims"])), name="mu")

        with tf.name_scope("distribution"):
            self.z = tf.placeholder(tf.float32, shape=(None, params["feature_dims"]))
            self.input_batch_size = tf.placeholder(tf.int32, shape=())

            self.q = self._soft_assignment(self.z, self.mu)
            self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster))
    
            self.pred = tf.argmax(self.q, axis=1)
        
        with tf.name_scope("dec-train"):
            self.loss = self._kl_divergence(self.p, self.q)
            self.optimizer = tf.train.AdamOptimizer(3e-3).minimize(self.loss)
            
    
    def get_assign_cluster_centers_op(self, features):
        # init mu
        print("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        print("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.
        
        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)
            
        Return:
            q_i_j: (num_points, num_cluster)
        """
        def _pairwise_euclidean_distance(a,b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.input_batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+1.0)/2.0)
        q = (q/tf.reduce_sum(q, axis=1, keepdims=True))
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p
    
    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target*tf.log(target/(pred)), axis=1))
    
    def cluster_acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
