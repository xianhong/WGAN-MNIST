import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import argparse 

parser=argparse.ArgumentParser()
parser.add_argument('meta_graph_appendix',nargs='?',
                    help="the appendix of meta graph to use",
                    type= int)
args=parser.parse_args()
meta_graph_file = "model.ckpt-"+str(args.meta_graph_appendix)+".meta"
 

log_dir= 'wgan'
z_dim=50
input_data_dir='mnist'

mnist = read_data_sets(input_data_dir,fake_data=False)
graph=tf.Graph()
sess = tf.Session(graph=graph)
# Load the WGAN model by restoring from the latest checkpoint file. 
with graph.as_default():
    saver = tf.train.import_meta_graph(
           os.path.join(log_dir, meta_graph_file))
    saver.restore(sess, tf.train.latest_checkpoint(log_dir))
    Z_ = tf.get_collection("Z_")[0]
    G_out = tf.get_collection("G_out")[0]

n = 15  # figure with 15x15 digits
digit_size = 28
# The feed_dict to be fed into WGAN model to get handwriting digit samples.
feed_dict = {Z_:np.random.uniform(size=(n*n,z_dim))}
# Generate n*n samples of handwriting digits using WGAN model.
samples = np.reshape(sess.run(G_out,feed_dict),(n,n,digit_size,digit_size))
figure = np.zeros((digit_size * n, digit_size * n))
# First display n*n sample digits.
for i in range(n):
    for j in range(n):
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = samples[j][i]

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')

# Then, randomly pick a starting position of MNIST training data set (total of 60K)
#  and display n*n handwriting digits starting from that position.
figure = np.zeros((digit_size * n, digit_size * n))
start_position = np.random.randint(0,60000-256)

for i in range(n):
    for j in range(n):
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = np.reshape(mnist.train.images[start_position+n*i+j],
                          (digit_size,digit_size))

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()        

sess.close()