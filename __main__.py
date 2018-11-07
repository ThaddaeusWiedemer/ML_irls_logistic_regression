import os
from sklearn.datasets import load_svmlight_file
import numpy as np
import tensorflow as tf

# ignore warning about missing AVX support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH_TRAIN = 'data/a9a'
PATH_TEST = 'data/a9a.t'
MAX_ITER = 100
LAMBDA = 0

# load all data
# the resulting X-vectors are transposed compared to their canonical form
X_TRAIN, Y_TRAIN = load_svmlight_file(PATH_TRAIN, n_features=123, dtype=np.float32)
X_TEST, Y_TEST = load_svmlight_file(PATH_TEST, n_features=123, dtype=np.float32)

# build the augmented vectors by stacking a column of 1's in front of X
# normally, this would be a row of 1's atop X, but here, x is transposed
N_TRAIN = X_TRAIN.shape[0]
N_TEST = X_TEST.shape[0]
X_TRAIN = np.hstack((np.ones((N_TRAIN, 1)), X_TRAIN.toarray()))
X_TEST = np.hstack((np.ones((N_TEST, 1)), X_TEST.toarray()))

# build true label vectors
Y_TRAIN = Y_TRAIN.reshape((N_TRAIN, 1))
Y_TEST = Y_TEST.reshape((N_TEST, 1))

# use 0 and 1 as labels instead of -1 and 1
Y_TRAIN = np.where(Y_TRAIN == -1, 0, 1)
Y_TEST = np.where(Y_TEST == -1, 0, 1)


def prob(x, w):
    """
    Returns the probability that a data point belongs to each class.

    Parameters
    ----------
    x: augmented feature vector of shape Nxd
    w: augmented weight vector of shape dx1

    Returns
    -------
    prob: probability for each class with shape N x num_classes(2)
    """
    y = tf.constant(np.array([0., 1.]), dtype=tf.float32)
    prob_ = tf.exp(tf.matmul(x, w) * y) / (1 + tf.exp(tf.matmul(x, w)))
    return prob_


def accuracy(x, y, w):
    """
    Returns the accuracy of the prediction.

    The accuracy is computed via accuracy = TP / (TP + FP) = TP / N.

    Parameters
    ----------
    x: augmented feature vector of shape Nxd
    y: true label vector of shape Nx1
    w: augmented weight vector of shape dx1

    Returns
    -------
    accuracy_: accuracy of the prediction
    """
    p = prob(x, w)

    # prediction is class with highest probability
    y_pred = tf.cast(tf.argmax(p, axis=1), tf.float32)
    y = tf.squeeze(y)

    # compute accuracy = TP / (TP + FP)
    accuracy_ = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
    return accuracy_


def update(w, x, y, l2_param=0):
    """
    Computes the weight update step as w_delta = (X'RX + \lambda * I)^(-1) (X' * (\mu - y) + \lambda * w_old).

    The new weight vector ic computed with w_new = w_old - w_delta.

    Parameters
    ----------
    w: current augmented weight vector of shape dx1
    x: augmented feature vector of shape Nxd
    y: true label vector of shape Nx1
    l2_param: for \lambda>0 use regularised log-likelihood

    Returns
    -------
    w_delta: weight update step of shape dx1
    """
    # get dimension d and current prediction mu of shape Nx1
    d = x.shape.as_list()[1]
    mu = tf.sigmoid(tf.matmul(x, w))

    # build R of shape Nx1 (element wise multiplication)
    r_flat = mu * (1 - mu)

    # build regularisation term and hessian H = X'RX of shape dxd
    l2_regularisation = l2_param * tf.eye(d)
    h = tf.matmul(tf.transpose(x), r_flat * x) + l2_regularisation

    # do single-value decomposition of H
    sigma, u, v = tf.svd(h, full_matrices=True, compute_uv=True)
    sigma = tf.expand_dims(sigma, 1)

    # calculate Moore-Penrose-pseudo-inverse of H via single value decomposition
    s_mppinv = tf.where(tf.not_equal(sigma, 0), 1 / sigma, tf.zeros_like(sigma))
    h_mppinv = tf.matmul(v, s_mppinv * tf.transpose(u))

    # calculate update step
    w_delta = tf.matmul(h_mppinv, tf.matmul(tf.transpose(x), mu - y) + l2_param * w)
    return w_delta


def optimise(w, w_delta):
    """
    Implements the update step.

    Parameters
    ----------
    w: current augmented weight vector of shape dx1
    w_delta: weight vector update step of shape dx1

    Returns
    -------
    updated w_old of shape dx1
    """
    return w.assign(w - w_delta)


def irls(x_train, y_train, x_test=None, y_test=None, l2_param=0, max_iter=MAX_ITER, verbose=False):
    """
    Estimates the parameters for logistic regression via Iterative Reweighted Least Squares algorithm (IRLS).

    Parameters
    ----------
    x_train: augmented feature vector of training data of shape Nxd
    y_train: true label vector of training data of shape Nx1
    x_test: augmented feature vector of test data of shape Nxd
    y_test: true label vector of training data of shape Nx1
    l2_param: for \lambda>0 use regularised log-likelihood
    max_iter: max number of iterations after which training will be stopped
    verbose: output more information

    Returns
    -------
    out: dict   'lambda'    value of l2 used
                'iteration' number of iterations trained
                'acc_train' accuracy in training
                'acc_test'  accuracy in testing
                'step'      last update step
    """
    # get dimensions and build representations of the data in the computational graph
    n, d = x_train.shape
    x = tf.placeholder(dtype=tf.float32, shape=(None, 124), name="x")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

    # initialise weight vector as matrix of 0.1 and set up node for update step, log-likelihood and accuracy
    w = tf.Variable(0.01*tf.ones((d, 1), dtype=tf.float32), name='w')
    w_delta = update(w, x, y, l2_param)
    with tf.variable_scope('accuracy'):
        acc = accuracy(x, y, w)

    # set up update operation node
    optimise_op = optimise(w, w_delta)

    # collect summaries
    tf.summary.scalar('acc', acc)
    merged_all = tf.summary.merge_all()

    # set up config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # run session and collect summary
    session = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter('./log', session.graph)
    session.run(tf.global_variables_initializer())

    # build feed dicts
    train_feed_dict = {x: x_train, y: y_train}
    test_feed_dict = {x: x_test, y: y_test}

    # generate console output
    if verbose:
        print('start training...')
        print('L2 param(lambda): {}'.format(l2_param))

    out = {'lambda': l2_param, 'iteration': 0, 'acc_train': 0, 'acc_test': 0, 'step': 0}

    i = 0
    # training loop
    while i <= max_iter:
        out['iteration'] = i
        if verbose:
            print('iteration {}'.format(i))

        # collect and print output
        acc_train, merged = session.run([acc, merged_all], feed_dict=train_feed_dict)
        summary_writer.add_summary(merged, i)
        acc_test = session.run(acc, feed_dict=test_feed_dict)
        l2_norm_w = np.linalg.norm(session.run(w))
        out['acc_train'] = acc_train
        out['acc_test'] = acc_test
        if verbose:
            print('\t train acc: {}, test acc: {}'.format(acc_train, acc_test))
            print('\t L2 norm of w: {}'.format(l2_norm_w))

        # print update step and stop if step is too small
        if i != 0:
            step = np.linalg.norm(session.run(w_delta, feed_dict=train_feed_dict))
            out['step'] = step
            if verbose:
                print('\t diff of w_old and w: {}'.format(step))
            if step < 1e-2:
                break

        session.run(optimise_op, feed_dict=train_feed_dict)
        i += 1
    if verbose:
        print('Finished training!')
    return out


# main
print('Lambda\tIterations\tTraining Accuracy\tTest Accuracy\t\tFinal Update Step')

# run and print output
output = irls(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, l2_param=LAMBDA, max_iter=MAX_ITER, verbose=True)
print('{lambda}\t\t{iteration}\t\t\t{acc_train}\t{acc_test}\t{step}'.format(**output))
