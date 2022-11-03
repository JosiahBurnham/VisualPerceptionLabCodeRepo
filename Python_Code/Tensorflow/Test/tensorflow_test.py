import tensorflow as tf
import cv2
import numpy

tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.compat.v1.Session()

# Evaluate the tensor `c`.
print(sess.run(c)) # prints 30.0

