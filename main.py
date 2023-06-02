import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import sionna

# Import required Sionna components
from sionna.fec.ldpc import LDPCBPDecoder
from sionna.utils.metrics import BitwiseMutualInformation
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples
from sionna.utils import ebnodb2no, hard_decisions
from sionna.utils.metrics import compute_ber
from sionna.utils.plotting import PlotBER

from tensorflow.keras.losses import BinaryCrossentropy


class WeightedBP(tf.keras.Model):
    def __init__(self, pcm, num_iter=5):
        super().__init__()

        # init components
        self.decoder = LDPCBPDecoder(pcm,
                                     num_iter=1,
                                     # iterations are done via outer loop (to access intermediate results for multi-loss)
                                     stateful=True,  # decoder stores internal messages after call
                                     hard_out=False,  # we need to access soft-information
                                     cn_type="boxplus",
                                     trainable=True)  # the decoder must be trainable, otherwise no weights are generated

        # used to generate llrs during training (see example notebook on all-zero codeword trick)
        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter

        self._bce = BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=1,  # BPSK
                              coderate=coderate)

        # all-zero CW to calculate loss / BER
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

        return c, c_hat, llr, loss


pcm_path = "BCH_alist/BCH_63_36_5_strip.alist.txt" # the path of parity check matrix
pcm_matrix = sionna.fec.utils.load_alist(pcm_path)
pcm, k, n, coderate = sionna.fec.utils.alist2mat(pcm_matrix)
num_iter = 5 # set number of decoding iterations

# and initialize the model
model = WeightedBP(pcm=pcm, num_iter=num_iter)

# training parameters
ebno_db = np.array(np.arange(4, 8, 1))  # s
batch_size = 1250
train_iter = 200
learning_rate = 0.01
clip_value_grad = 10 # gradient clipping for stable training convergence

# bmi is used as metric to evaluate the intermediate results
bmi = BitwiseMutualInformation()

# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
for i in range(0, ebno_db.size):
    for it in range(0, train_iter):
        with tf.GradientTape() as tape:
            c, c_hat, llr, loss = model(batch_size, ebno_db[i])
            mask = np.ones(batch_size, dtype=bool)
            for j in range(0, c.shape[0]):
                d_in = sionna.utils.count_block_errors(hard_decisions(llr[j]), c[j])
                d_out = sionna.utils.count_block_errors(c_hat[j], c[j])
                if d_out == 0 or d_out >= d_in:
                    mask[j] = False
            c = tf.boolean_mask(c, mask)
            c_hat = tf.boolean_mask(c_hat, mask)
            llr = tf.boolean_mask(llr, mask)
        grads = tape.gradient(loss, model.trainable_variables)
        grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # calculate and print intermediate metrics
    # only for information
    # this has no impact on the training
        if it % 10 == 0:  # evaluate every 10 iterations
            # calculate ber from received LLRs
            # b_hat = hard_decisions(llr)  # hard decided LLRs first
            ber = compute_ber(c, c_hat)
            # and print results
            mi = bmi(c, llr).numpy()  # calculate bit-wise mutual information
            l = loss.numpy()  # copy loss to numpy for printing
            print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
            bmi.reset_states()  # reset the BMI metric