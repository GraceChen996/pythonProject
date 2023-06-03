import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import sionna
# Import required Sionna components
from sionna.fec.ldpc import LDPCBPDecoder, LDPC5GEncoder, LDPC5GDecoder
from sionna.utils.metrics import BitwiseMutualInformation
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples
from sionna.utils import ebnodb2no, hard_decisions
from sionna.utils.metrics import compute_ber
from sionna.utils.plotting import PlotBER

from tensorflow.keras.losses import BinaryCrossentropy


class WeightedBP(keras.Model):
    def __init__(self, pcm, num_iter=5):
        # init components
        super().__init__()
        self.decoder = LDPCBPDecoder(pcm,
                                     num_iter=1,
                                     # iterations are done via outer loop (to access intermediate results for multi-loss)
                                     stateful=True,  # decoder stores internal messages after call
                                     hard_out=False,  # we need to access soft-information
                                     cn_type="boxplus",
                                     trainable=True)  # the decoder must be trainable, otherwise no weights are generated

        # used to generate llrs during training (see example notebook on all-zero codeword trick)
        self.llr_source = sionna.fec.utils.GaussianPriorSource()
        self._num_iter = num_iter
        self._bce = BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=1,  # BPSK
                              coderate=coderate)

        # all-zero CW to calculate loss / BE
        c = tf.zeros([batch_size, n])

        # Gaussian LLR source
        llr = self.llr_source([[batch_size, n], noise_var])

        msg_vn = None  # internal state of decoder
        for i in range(self._num_iter):
            x_hat, msg_vn = self.decoder((llr, msg_vn))  # perform one decoding iteration; decoder returns soft-values

        return c, x_hat, llr
        # --- implement multi-loss as proposed by Nachmani et al. [1]---
        # loss = 0
        # msg_vn = None  # internal state of decoder
        # for i in range(self._num_iter):
        #     x_hat, msg_vn = self.decoder((llr, msg_vn))  # perform one decoding iteration; decoder returns soft-values
        #     loss += self._bce(c, x_hat)  # add loss after each iteration
        #
        # loss /= self._num_iter  # scale loss by number of iterations
        # return c, x_hat, llr


def choosePrior(S, c):

    model = WeightedBP(pcm=pcm, num_iter=num_iter)
    c, x_hat, llr = model(batch_size, ebno_db[i])
    abp = np.abs(c - llr) / n
    mbce = np.abs(c * np.log(llr) + (1 - c) * np.log(1 - llr)) / n


pcm_path = "BCH_alist/BCH_63_36_5_strip.alist.txt"  # the path of parity check matrix
pcm_matrix = sionna.fec.utils.load_alist(pcm_path)
pcm, k, n, coderate = sionna.fec.utils.alist2mat(pcm_matrix)
num_iter = 5  # set number of decoding iterations

# initialize the model
model = WeightedBP(pcm=pcm, num_iter=num_iter)

# SNR to simulate the results
ebno_db = np.array(np.arange(4, 8, 1))  # s
# ebno_db = np.array(4)
batch_size = 1250
train_iter = 200
learning_rate = 0.01
clip_value_grad = 10  # gradient clipping for stable training convergence

# Training
# bmi is used as metric to evaluate the intermediate results
bmi = BitwiseMutualInformation()

# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
bce = BinaryCrossentropy(from_logits=True)

mu, sigma = choosePrior()
for i in range(0, ebno_db.size):
    print('SNR={}'.format(ebno_db[i]))
    for it in range(0, train_iter):
        loss = 0
        with tf.GradientTape() as tape:
            c, x_hat, llr = model(batch_size, ebno_db[i])
            abp = np.abs(c - llr) / n
            mbce = np.abs(c * np.log(llr) + (1 - c) * np.log(1 - llr)) / n
            theta = tf.concat(abp, mbce)
            # --- implement multi-loss as proposed by Nachmani et al. [1]---
            for ind in range(num_iter):
                # temp = tf.cast(sionna.utils.hard_decisions(x_hat), tf.float32)
                loss += bce(c, x_hat)  # add loss after each iteration
            loss /= num_iter  # scale loss by number of iterations

        grads = tape.gradient(loss, model.trainable_variables)
        grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if it % 10 == 0:  # evaluate every 10 iterations
            ber = compute_ber(c, hard_decisions(x_hat))
            # and print results
            mi = bmi(c, llr).numpy()  # calculate bit-wise mutual information
            l = loss.numpy()  # copy loss to numpy for printing
            print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
            bmi.reset_states()  # reset the BMI metric

ebno_dbs = np.array(np.arange(1, 7, 0.5))
batch_size = 10000
mc_iters = 100
ber_plot = PlotBER("Weighted BP")
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000,  # stop sim after 2000 bit errors
                  legend="Trained",
                  max_mc_iter=mc_iters,
                  soft_estimates=True);
