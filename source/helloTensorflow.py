import tensorflow as tf
deep_learning = tf.constant('Deep Learning')
session = tf.Session()

weights = tf.Variable(tf.random_normal([300, 200], stddev=0.5), name="weights")
print(session.run(deep_learning))
