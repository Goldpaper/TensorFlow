#11주차 실습

import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

print(X)
print(Y)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#learning rate = 알파 값 (처음 시작했을 때는 0.1 사람이 지정하면 알아서 프로그램이 적잘한 값을 찾음

train_op = optimizer.minimize(cost)

#세션을 열어
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in  range(100) :
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y:y_data})
        #under_bar는 리턴값이 2개인데 안쓰는 값을 그냥 언더바로 표시

        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n===TEST===")
    print("X: 5, Y:",sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:",sess.run(hypothesis, feed_dict={X: 2.5}))