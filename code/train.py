import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)
def Conv(img, dropout, num_class):
  print(img.shape)
  conv1 = tf.layers.conv2d(img, 32, 5, activation=tf.nn.leaky_relu)
  print(conv1.shape)
  # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
  conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
  print(conv1.shape)
  # Convolution Layer with 64 filters and a kernel size of 3
  conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.leaky_relu)
  print(conv2.shape)
  # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
  conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
  print(conv2.shape)
  # Flatten the data to a 1-D vector for the fully connected layer
  fc1 = tf.contrib.layers.flatten(conv2)
  print(fc1.shape)
  # Fully connected layer (in tf contrib folder for now)
  fc1 = tf.layers.dense(fc1, 1024)
  print(fc1.shape)
  # Apply Dropout (if is_training is False, dropout is not applied)
  fc1 = tf.layers.dropout(fc1, rate=dropout)
  # Output layer, class prediction
  out = tf.layers.dense(fc1, num_class)
  print(out.shape)
  return out
with tf.variable_scope('input'):
  X = tf.placeholder(tf.float32,shape=[None,784])
with tf.variable_scope('output'):
  Y = tf.placeholder(tf.float32,shape=[None,10])

img = tf.reshape(X,shape=[-1,28,28,1])
result = Conv(img, 0.3, 10)

with tf.variable_scope('loss'):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,labels=Y))
  prediction_prob = tf.nn.softmax(result)

with tf.variable_scope('trainer'):
  trainer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.variable_scope('logging'):
  tf.summary.scalar('current_cost',loss)
summary = tf.summary.merge_all()
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  training_writer = tf.summary.FileWriter('log/training', sess.graph)
  for epoch in range(10000):
     batch_x, batch_y = mnist.train.next_batch(100)
     sess.run([trainer],feed_dict={X:batch_x,Y:batch_y})
     if epoch%100==0:
       error,summ = sess.run([loss,summary],feed_dict={X:batch_x,Y:batch_y})
       training_writer.add_summary(summ, 10000)
  print("Training Complete")
  final_training_cost = sess.run(loss, feed_dict={X: batch_x, Y: batch_y})
  print("Final training cost {}".format(final_training_cost))
  saver.save(sess, "log/model.ckpt")
