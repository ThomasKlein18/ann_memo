import numpy as np 
import tensorflow as tf 
import sys

from tensorflow.keras import layers 


np.set_printoptions(threshold=sys.maxsize)

def num_to_binary(num, digits):
    """
    Converts a given integer to a binary representation of that integer. Second parameter not yet used.
    """
    #return list('{0:02b}'.format(num))
    return list(str('{0:0'+str(digits)+'b}').format(num))



if __name__ == "__main__":

    ## paths ##
    storage_path = "../"
    plot_path = storage_path + "plots/"
    weight_path = storage_path + "weights/"

    ## parameters ##
    max_samples = 2
    out_size = 2

    assert max_samples < 2**out_size, "Encoding {} samples into {} binary digits is not possible, limit is {}.".format(max_samples, out_size, 2**out_size)

    ## model definition ##
    model = tf.keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='tanh', input_shape=(784,))) #starting with MNIST / fashionMNIST
    #model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(out_size, activation='sigmoid'))

    ## model setup ##
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                loss='mean_squared_error',
                metrics=['accuracy'])

    ## get dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images[0:max_samples,:,:]
    train_labels = train_labels[0:max_samples]

    # replacing labels with numbers
    train_labels = np.array([num_to_binary(i,out_size) for i in range(len(train_labels))])

    


    callbacks = [
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir=plot_path)
    ]


    model.fit(train_images,
                train_labels,
            epochs=10000,
            batch_size=1,
            #validation_data=(test_images,test_labels),
            #validation_steps=100,
            callbacks = callbacks)

    model.save_weights(weight_path)

    #TODO: print some model predictions, see what happens
