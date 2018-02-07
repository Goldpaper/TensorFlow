import tensorflow as tf

hello = tf.constant('Hello world!')
a = tf.constant(10)

sess = tf.Session()

print(sess.run(hello))
print(sess.run(a))

sess.close()