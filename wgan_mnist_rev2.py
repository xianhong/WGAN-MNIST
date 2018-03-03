import os
import tensorflow as tf
import tensorflow.contrib.layers as ly
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import argparse 

parser=argparse.ArgumentParser()
parser.add_argument('meta_graph_appendix',nargs='?',
                    help="the appendix of meta graph,if recovering from checkpoint",
                    type= int)
parser.add_argument('--force','-f',dest='restore',action='store_false',default=True,
                    help="force to train the model from scratch")
args=parser.parse_args()
print (args.restore)
restore = args.restore
if restore:
    if args.meta_graph_appendix==None:
        print ("missing the appendix number of meta graph file to recover from.")
    else:
        meta_graph_file = "model.ckpt-"+str(args.meta_graph_appendix)+".meta"
        print ("Restoring from:",meta_graph_file)
else:
    print ("Training the model from scrtach.")


max_steps=20000
log_dir= 'wgan'
z_dim=50
batch_size=128
# restore==True,if we resume model training from previously saved checkpoint file.
# Otherwise, we're training the model for the first time.
#restore=True    
learning_rate=1e-4     # Learning rate for the discriminator.
learning_rate2=1e-4    # learning rate for the generator.
input_data_dir='mnist'


def sample(n):
  z = tf.random_uniform([n,z_dim])
  h = generator(z, reuse=True)
  tf.summary.image('image', h, max_outputs=n)
 
def lrelu(x, leak=0.3, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
  return f1 * x + f2 * abs(x)
 
def generator(z, reuse=False): 
  nch = 256
  with tf.variable_scope('G', reuse=reuse):
    h = tf.layers.dense(z, 3*3*nch, kernel_initializer=tf.random_normal_initializer())
    h = lrelu(h)
    h = tf.reshape(h, [-1,3,3,nch])
    h = tf.layers.conv2d_transpose(h, nch//2, 3, strides=2,
                  padding='valid', activation=lrelu)
    h = tf.layers.conv2d_transpose(h, nch//4, 3, strides=2,
                  padding='same', activation=lrelu)
    h = tf.layers.conv2d_transpose(h, nch//8, 3, strides=2,
                  padding='same')
    h = tf.layers.conv2d(h, 1, 1)
    h = tf.sigmoid(h)
  return h
 
def discriminator(h, reuse=False):
  with tf.variable_scope('D', reuse=reuse):
    size = 32
    h = ly.conv2d(h, num_outputs=size, kernel_size=3,
                  stride=2, activation_fn=lrelu)
    h = ly.conv2d(h, num_outputs=size * 2, kernel_size=3,
                  stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                  normalizer_params={'is_training':True})
    h = ly.conv2d(h, num_outputs=size * 4, kernel_size=3,
                  stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm, 
                  normalizer_params={'is_training':True})
 
    h = ly.conv2d(h, num_outputs=size * 8, kernel_size=3,
                  stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                  normalizer_params={'is_training':True})
    h = ly.fully_connected(tf.reshape(
                  h, [batch_size, -1]), 1, activation_fn=None)
  return h
 
def build_graph():
      X = tf.placeholder(tf.float32, shape=(batch_size,28*28))
      X_ = tf.reshape(X, [-1,28,28,1])
      # Z_ is the input for the model to generate handwriting digts ; 
      # G_out is what the model spits out
      Z_ = tf.placeholder(tf.float32,shape=(None,z_dim))
      Z = tf.random_uniform([batch_size,z_dim])
      global_step = tf.Variable(0, trainable=False, name='global_step')
      D_real = discriminator(X_)
      D_fake = discriminator(generator(Z), reuse=True)
     
      sample(8) # for Tensorboard visualization
      G_out = generator(Z_,reuse=True)
      D_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D')
      G_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
     
      D_loss = tf.reduce_mean(D_fake - D_real)
      G_loss = tf.reduce_mean(- D_fake)
     
      tf.summary.scalar('D_loss', D_loss)
      tf.summary.scalar('G_loss', G_loss)
     
      D_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(D_loss, var_list=D_weights)
      G_solver = tf.train.RMSPropOptimizer(learning_rate2).minimize(G_loss, 
                                          global_step=global_step,var_list=G_weights)
      clip_weights = [tf.assign(w,tf.clip_by_value(w, -0.01, 0.01)) for w in D_weights] 
      return X, D_solver, G_solver, clip_weights, D_loss, G_out, Z_
 
mnist = read_data_sets(input_data_dir,fake_data=False)
graph=tf.Graph()
sess = tf.Session(graph=graph)

# If we're training the model for the first time.
if restore==False: 
    with graph.as_default():
        X, D_s, G_s, clip, D_l, G_out,Z_ = build_graph()
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=3)
        # Add these values (tensors) to collection to facilitate model restore.
        tf.add_to_collection("X", X)
        tf.add_to_collection("D_s", D_s)
        tf.add_to_collection("G_s", G_s)
        for t in clip:
            tf.add_to_collection("clip", t)
        tf.add_to_collection("Z_", Z_)
        tf.add_to_collection("G_out", G_out)
        tf.add_to_collection("summary", summary)
        tf.add_to_collection("D_l",D_l)      
    # After building the graph, we save the graph .
    summary_writer = tf.summary.FileWriter(log_dir, graph)   
    sess.run(init)

# If we have a previsouly saved checkpoint, we could continue on training the model. 
if restore:
    with graph.as_default():
        saver = tf.train.import_meta_graph(
                os.path.join(log_dir, meta_graph_file))
        saver.restore(sess, tf.train.latest_checkpoint(log_dir))
        # Get values (tensors) from the collection for data feed-in and tensors evaluation. 
        X = tf.get_collection("X")[0]
        D_s = tf.get_collection("D_s")[0]
        G_s = tf.get_collection("G_s")[0]
        clip = tf.get_collection("clip")
        Z_ = tf.get_collection("Z_")[0]
        G_out = tf.get_collection("G_out")[0]
        summary = tf.get_collection("summary")[0]
        D_l = tf.get_collection("D_l")[0]
        summary_writer = tf.summary.FileWriter(log_dir, graph)
        
step = sess.run("global_step:0")
 
print ('START TRAINING at step:',step)
while step < max_steps:
    for _ in range(5):
        feed_dict = {X: mnist.train.next_batch(batch_size)[0]}
        sess.run(D_s, feed_dict=feed_dict)
        sess.run(clip)
    sess.run(G_s)
 
    if step % 50 == 0:
        summary_str, D_loss = sess.run([summary, D_l], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        print ('STEP:', step, '\tDLOSS', D_loss)
 
    if (step + 1) % 500 == 0 or (step + 1) == max_steps:
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=sess.run("global_step:0"))
    step += 1
sess.close()
