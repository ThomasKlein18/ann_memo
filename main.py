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

def binary_estimates_to_num(binary):
    return np.sum([j * 2**i for i,j in enumerate(reversed(np.round(binary)))])

if __name__ == "__main__":

    ## paths ##
    storage_path = "../"
    version = "v0/"
    plot_path = storage_path + version + "plots/"
    weight_path = storage_path + version + "weights/"


   

    """
    A remark on the loss: 
    sparse crossentropy just means: We get the label as an integer 
    between 0 and 9. categorical crossentropy means that we expect
    one-hot encoded labels.

    If you use categorical crossentropy to train on memorized labels, 
    all of the outputs just go to 1, so that is why you see a stagnating loss
    and a slightly larger accuracy (due to the way accuracy is calculated.)
    """
    ## parameters ##
    used_loss = 'mean_squared_error' #'categorical_crossentropy'
    max_samples = 1000
    out_size = 16
    enable_memorization = True

     ## network parameters ##
    num_epochs = max_samples * 100
    batchsize = 100

    assert batchsize <= max_samples, "Batch size can't be bigger than number of samples."

    # automatically adjusting out_size and loss function if we are not memorizing
    out_size = 10 if not enable_memorization else out_size
    #used_loss = 'sparse_categorical_crossentropy' if not enable_memorization else 'categorical_crossentropy'
    if enable_memorization:
        assert max_samples < 2**out_size, "Encoding {} samples into {} binary digits is not possible, limit is {}.".format(max_samples, out_size, 2**out_size)

    ## model definition ##
    model = tf.keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='tanh', input_shape=(784,))) #starting with MNIST / fashionMNIST
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(out_size, activation='sigmoid'))

    ## model setup ##
    model.compile(optimizer=tf.train.AdagradOptimizer(0.005),
                loss=used_loss,
                metrics=['accuracy'])

    # getting the dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # only using the first max_samples samples
    train_images = train_images[0:max_samples,:,:]
    train_labels = train_labels[0:max_samples]

    if enable_memorization :
        # replacing labels with numbers
        train_labels = np.array([num_to_binary(i,out_size) for i in range(len(train_labels))])


    callbacks = [
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir=plot_path)
    ]

    model.fit(train_images,
            train_labels,
            epochs=num_epochs,
            batch_size=batchsize,
            #validation_data=(test_images,test_labels),
            #validation_steps=100,
            callbacks = callbacks)

    model.save_weights(weight_path)

    positives = 0
    for i in range(max_samples):
        test_img = train_images[i,:,:]
        test_img = np.reshape(test_img, [1,28,28])
        pred = model.predict(test_img, batch_size=1)[0]
        
        if(binary_estimates_to_num(pred) == i):
            positives += 1
    print("Result:",positives/max_samples)
