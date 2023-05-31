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
                                     hard_out=True,  # we need to access soft-information
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

        # --- implement multi-loss as proposed by Nachmani et al. [1]---
        loss = 0
        msg_vn = None  # internal state of decoder
        for i in range(self._num_iter):
            c_hat, msg_vn = self.decoder((llr, msg_vn))  # perform one decoding iteration; decoder returns soft-values
            loss += self._bce(c, c_hat)  # add loss after each iteration

        loss /= self._num_iter  # scale loss by number of iterations

        return c, c_hat, loss


pcm_path = "BCH_alist/BCH_63_36_5_strip.alist.txt"  # the path of parity check matrix
pcm_matrix = sionna.fec.utils.load_alist(pcm_path)
pcm, k, n, coderate = sionna.fec.utils.alist2mat(pcm_matrix)
num_iter = 5  # set number of decoding iterations

# and initialize the model
model = WeightedBP(pcm=pcm, num_iter=num_iter)

# SNR to simulate the results
ebno_dbs = np.array(np.arange(1, 10.5, 0.5))  # s
mc_iters = 100  # number of Monte Carlo iterations
batch_size = 1000
for i in range(0, ebno_dbs.size):
    error = np.zeros(1000)
    c, c_hat, loss = model.call(batch_size, ebno_dbs[i])
    for j in range(0, batch_size):
        error[j] = sionna.utils.count_block_errors(c[j], c_hat[j])
    dH = np.min(error)
