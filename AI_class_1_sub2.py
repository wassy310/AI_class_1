model.save("./drive/MyDrive/mnist_model.h5")

import tensorflow as tf

def load_imgset(filename):
    img = tf.io.decode_image(tf.io.read_file(filename))
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [28, 28])
    img / 255.0
    img = np.reshape(img, (28, 28,1))
    img_set = np.expand_dims(img,0)
    return img_set

model = tf.keras.models.load_model("./drive/MyDrive/mnist_model.h5")

for i in range(10):
  img = load_imgset("./drive/MyDrive/Mynum/%d.png" % i)
  pred = model.predict(img)
  pred_num = pred.argmax()
  print(i, "-->", pred_num)
