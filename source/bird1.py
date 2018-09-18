import tensorflow as tf
import numpy as np 

x_data = np.array([
    [0, 0], 
    [1, 0], 
    [1, 1], 
    [0, 0], 
    [0, 0], 
    [0, 1]])

y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

# [6, 2] * [2, 10] = [6, 10]
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# [6, 10] * [10, 3] = [6, 3]
model = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model)
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('predition: ', sess.run(prediction, feed_dict={X: x_data}))
print('target: ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))



