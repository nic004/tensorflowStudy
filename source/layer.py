import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(y, feed_dict={x: [[1, 2, 3]]}))