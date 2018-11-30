"""
Want to quickly try to replicate [Learning Game of Life with a Convolutional
Neural Network](https://danielrapp.github.io/cnn-gol/). Seems like we should
be able to create an exactly correct network by hand with many fewer weights.
"""

from types import SimpleNamespace
import tensorflow as tf
import numpy as np
import scipy.signal


class LifeNetwork(object):
    def __init__(self):
        tf.reset_default_graph()
        op = self.op = SimpleNamespace()
        op.input = tf.placeholder(tf.uint8, [None, None])
        op.test = tf.placeholder(tf.uint8, op.input.shape)
        in_data = tf.cast(op.input, tf.float32)
        in_data = tf.expand_dims(in_data, 0)
        in_data = tf.expand_dims(in_data, -1)
        op.hidden = tf.layers.conv2d(
            in_data, filters=2, kernel_size=3, activation=tf.tanh, padding='same')
        op.logit = tf.layers.conv2d(
            op.hidden, filters=1, kernel_size=1, padding='same')[0,:,:,0]
        op.output = tf.cast(op.logit > 0, tf.uint8)
        op.accuracy = tf.cast(tf.equal(op.test, op.output), tf.float32)
        op.accuracy = tf.reduce_mean(op.accuracy)
        correct_logit = op.logit * (2*tf.cast(op.test, tf.float32) - 1)
        # loss = -log(1/(1+e^-logit)) = log(1+e^-logit)
        # log(1+e^-x) = log(e^-x) + log(e^x+1)
        op.loss = tf.log(1+tf.exp(-tf.abs(correct_logit)))
        op.loss -= correct_logit * tf.cast(correct_logit < 0.0, tf.float32)
        op.loss = tf.reduce_mean(op.loss)
        op.train = tf.train.AdamOptimizer().minimize(op.loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def exact_evolve_board(self, board):
        neighbors = scipy.signal.convolve(board, [[1,1,1],[1,0,1],[1,1,1]], 'same')
        board = (neighbors == 3) | ((board == 1) & (neighbors == 2))
        return board.astype(np.uint8)

    def network_evolve_board(self, board):
        op = self.op
        board = self.session.run(op.output, {op.input: board.astype(np.float32)})
        return board.astype(np.uint8)

    def hand_tune_network(self):
        """
        Hand pick weights to get 100% accuracy.
        """
        var_w1, var_w2 = tf.get_collection('variables', 'conv2d.*/kernel:0')
        var_b1, var_b2 = tf.get_collection('variables', 'conv2d.*/bias:0')
        eps = 1e-2
        w1 = np.zeros(var_w1.shape.as_list(), dtype=np.float32)
        w1[...,:2] = 1
        w1[1,1,0,1] = 0
        b1 = np.zeros(w1.shape[-1], dtype=np.float32)
        # In order to be alive in the next generation, a cell needs to
        # have the sum of its predecessor and all neighbors to be at least 3,
        # (given by first filter), and the sum of all its neighbors to be less
        # than 4 (second filter).
        b1[:2] = [-2.5, -3.5]
        var_w1.load(w1/eps, self.session)
        var_b1.load(b1/eps, self.session)
        # The second layer is active when the first filter fires but the second
        # does not.
        w2 = np.zeros(var_w2.shape.as_list(), dtype=np.float32)
        w2[0,0,:2,0] = [1, -1]
        var_w2.load(w2/eps, self.session)
        var_b2.load(np.array([-1/eps], dtype=np.float32), self.session)

    def train(self, size=(20, 20), num_steps=1+10**4, report_every=10**2):
        """
        This usually gets to 100% accuracy in about 7000 steps when started
        from random initialization.
        """
        op = self.op
        for step in range(num_steps):
            board = np.random.randint(2, size=size, dtype=np.uint8)
            test = self.exact_evolve_board(board)
            accuracy, loss, _ = self.session.run(
                [op.accuracy, op.loss, op.train],
                {op.input: board, op.test: test})
            if step % report_every == 0:
                print(f"{step}: acc={100*accuracy:0.2f}% loss={loss:0.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--handtune', action="store_true")
    args = parser.parse_args()
    model = LifeNetwork()
    if args.handtune:
        model.hand_tune_network()
    model.train()
