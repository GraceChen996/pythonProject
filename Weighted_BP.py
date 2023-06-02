import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
# Import Sionna
import sionna
# Import required Sionna components
from sionna.fec.ldpc import LDPCBPDecoder, LDPC5GEncoder, LDPC5GDecoder
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

        return c, c_hat, loss

pcm_path = "BCH_alist/BCH_63_36_5_strip.alist.txt" # the path of parity check matrix
pcm_matrix = sionna.fec.utils.load_alist(pcm_path)
pcm, k, n, coderate = sionna.fec.utils.alist2mat(pcm_matrix)
num_iter = 5 # set number of decoding iterations

# and initialize the model
model = WeightedBP(pcm=pcm, num_iter=num_iter)

# SNR to simulate the results
ebno_dbs = np.array(np.arange(1, 10.5, 0.5))
mc_iters = 100 # number of Monte Carlo iterations

# we generate a new PlotBER() object to simulate, store and plot the BER results
ber_plot = PlotBER("Weighted BP")

# simulate and plot the BER curve of the untrained decoder
ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=1000, # stop sim after 1000 bit errors
                  legend="classical BP decoder",
                  soft_estimates=True,
                  max_mc_iter=mc_iters,
                  forward_keyboard_interrupt=False);

# training parameters
batch_size = 1000
train_iter = 200
ebno_db = 4.0
clip_value_grad = 10 # gradient clipping for stable training convergence

# bmi is used as metric to evaluate the intermediate results
bmi = BitwiseMutualInformation()

# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-2)

for it in range(0, train_iter):
    with tf.GradientTape() as tape:
        b, llr, loss = model(batch_size, ebno_db)

    grads = tape.gradient(loss, model.trainable_variables)
    grads = tf.clip_by_value(grads, -clip_value_grad, clip_value_grad, name=None)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # calculate and print intermediate metrics
    # only for information
    # this has no impact on the training
    if it%10==0: # evaluate every 10 iterations
        # calculate ber from received LLRs
        b_hat = hard_decisions(llr) # hard decided LLRs first
        ber = compute_ber(b, b_hat)
        # and print results
        mi = bmi(b, llr).numpy() # calculate bit-wise mutual information
        l = loss.numpy() # copy loss to numpy for printing
        print(f"Current loss: {l:3f} ber: {ber:.4f} bmi: {mi:.3f}".format())
        bmi.reset_states() # reset the BMI metric

ebno_dbs = np.array(np.arange(1, 10, 0.5))
batch_size = 10000
mc_ites = 100

ber_plot.simulate(model,
                  ebno_dbs=ebno_dbs,
                  batch_size=1000,
                  num_target_bit_errors=2000, # stop sim after 2000 bit errors
                  legend="Trained",
                  max_mc_iter=mc_iters,
                  soft_estimates=True);