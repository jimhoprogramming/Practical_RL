# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

@ tf.function
def define_nn():
    nn = tf.keras.Sequential([
                Conv2D(filters = 16, kernel_size = (3,3), strides = (2, 2), activation = 'relu'),
                Conv2D(filters = 32, kernel_size = (3,3), strides = (2, 2), activation = 'relu'),
                Conv2D(filters = 64, kernel_size = (3,3), strides = (2, 2), activation = 'relu'),
                Dense(units = 256, activation = 'relu')
            ])
    base_learning_rate = 0.001
    nn.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                       loss='sparse_categorical_crossentropy',
                       metrics=['sparse_categorical_accuracy'])
    state_t =  tf.zeros(shape=(4, 64, 64, 1))# 自己见一个模拟图像
    #qvalues_t = nn.predict(x = state_t, steps  = 1)
    nn.summary()

def test_using_FAPI():
    inputs = tf.keras.Input(shape = (64,64,1), name = 'game_img')
    x = Conv2D(filters = 16, kernel_size = (3,3), strides = (2, 2), activation = 'relu')(inputs)
    x = Conv2D(filters = 32, kernel_size = (3,3), strides = (2, 2), activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = (2, 2), activation = 'relu')(x)
    x = Flatten()(x)
<<<<<<< HEAD
    outputs = Dense(units = 4, activation = 'relu')(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'nn')
    model.summary()
=======
    outputs = Dense(units = 256, activation = 'relu')(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'nn')
    #model.summary()
>>>>>>> 87a73886a358aa4bc0dae10efa7c7aabca3a70ac
    #tf.keras.utils.plot_model(model, 'nn.png', show_shapes = True)
    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                       loss='sparse_categorical_crossentropy',
                       metrics=['sparse_categorical_accuracy'])
    state_t =  tf.zeros(shape=(4, 64, 64, 1))# 自己见一个模拟图像
    predict = model.predict(state_t)
    print(predict.shape)
<<<<<<< HEAD
    print('redict = {}'.format(predict))
    #
    label = tf.ones((4))
    rel = model.evaluate(state_t, label)
    print(rel)
    
if __name__=='__main__':
    #define_nn()
    test_using_FAPI()
=======
if __name__=='__main__':
    define_nn()
    #test_using_FAPI()
>>>>>>> 87a73886a358aa4bc0dae10efa7c7aabca3a70ac
