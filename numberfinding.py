import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=3)

loss, accuracy = model.evaluate(x_test,y_test)
print(accuracy)
print(loss)
model.save('digits.model')
array_list = []
for x in range(0,10):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is prebably: {np.argmax(prediction)}')
    a = np.argmax(prediction)
    array_list.append(a)
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
print(array_list)
x = np.array(array_list)
y = np.array([0,1,2,3,4,5,6,7,8,9])
plt.scatter(x, y)
plt.show()