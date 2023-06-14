import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
# Import Sionna
import sionna
import WeightedBP
# Import required Sionna components
from sionna.fec.ldpc import LDPCBPDecoder, LDPC5GEncoder, LDPC5GDecoder
from sionna.utils.metrics import BitwiseMutualInformation
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples
from sionna.utils import ebnodb2no, hard_decisions
from sionna.utils.metrics import compute_ber
from sionna.utils.plotting import PlotBER

from tensorflow.keras.losses import BinaryCrossentropy

pcm_path = "BCH_alist/BCH_127_106_3_strip.alist.txt"  # the path of parity check matrix
pcm_matrix = sionna.fec.utils.load_alist(pcm_path)
pcm, k, n, coderate = sionna.fec.utils.alist2mat(pcm_matrix)
num_iter = 5  # 译码迭代次数,和隐藏层数量有关
mc_iters = 100  # number of Monte Carlo iterations
# and initialize the model
model = WeightedBP.WeightedBP(coderate, n, pcm=pcm, num_iter=num_iter)
ber_plot = PlotBER(title="Weighted BP")
simulate_ebno_dbs = np.array(np.arange(1, 10, 1))
training_ebno_dbs = np.array(np.arange(3, 6, 1))

# 传统BP仿真----------------------------
WeightedBP.classicalBP(ber_plot, model, mc_iters,simulate_ebno_dbs)
ber_plot(show_ber=True,save_fig=True,path='result/classicalBP')

# # training--------------------------------------
mini_batch_size = 120
train_iter = 100
clip_value_grad = 10  # gradient clipping for stable training convergence
learning_rate = 1e-3
# bmi is used as metric to evaluate the intermediate results
bmi = BitwiseMutualInformation()

# try also different optimizers or different hyperparameters
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
WeightedBP.training(train_iter, optimizer, model, mini_batch_size, training_ebno_dbs, clip_value_grad, bmi)
#
# # 权重BP---------------------------------------
WeightedBP.trainedBP(ber_plot, model, mc_iters, simulate_ebno_dbs)
ber_plot(show_ber=True, save_fig=True, path='result/weightedBP')
