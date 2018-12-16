"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""
import os
import cv2
import glob
import ntpath
import random
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.layers import Lambda
from keras import layers, models, optimizers, losses
from keras import backend as K
from keras.layers import Lambda
import matplotlib.pyplot as plt
from utils import combine_images
from keras.utils import to_categorical
from keras import layers, models, optimizers
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K.set_image_data_format('channels_last')
mean = -1          # Dummy Values
log_variance = -1  # Dummy Values

# Change this dataset
dataset_path = "../Dataset/Animals/"

def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    global mean
    mean = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='mean')(primarycaps)
    global log_variance
    log_variance = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='log_variance')(primarycaps)

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(variance)*eps
    def reparam(args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        
        mean, log_variance = args
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=K.shape(mean))
        return (mean + K.exp(0.5 * log_variance) * epsilon)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(reparam, name='z')([mean, log_variance])
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(z)

    # Hierarchy output section
    #------------------------------------------------------------------------------------------------------------------------------
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([z, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(z)  # Mask using the capsule with maximal length. For prediction

    '''
    def longest_vector_retrieve(args):
    	z, out_caps = args
    	only_length_vector = tf.transpose(tf.nn.embedding_lookup(tf.transpose(out_caps),[1]))
    	only_length_vector = tf.reshape(only_length_vector,[-1])
    	longest_vector_index = tf.argmax(only_length_vector, axis=-1)
    	print(longest_vector_index)
    	return tf.gather_nd(z,[longest_vector_index])
    longest_vector = Lambda(longest_vector_retrieve,name = 'longest_vector')([z,out_caps])
    '''
    longest_vector_train = masked_by_y
    longest_vector_eval = masked
    # Keep adding hierarchies
    # Face hierarchy
    face = layers.Dense(units=5,name='face')
    face_train = face(longest_vector_train)
    face_eval = face(longest_vector_eval)
    eyes = layers.Dense(units=1,name='eyes')
    eyes_train = eyes(face_train)
    eyes_eval = eyes(face_eval)
    mouth_open = layers.Dense(units=1,name='mouth_open')
    mouth_open_train = mouth_open(face_train)
    mouth_open_eval = mouth_open(face_eval)
    snout = layers.Dense(units=1,name='snout')
    snout_train = snout(face_train)
    snout_eval = snout(face_eval)
    ears = layers.Dense(units=1,name='ears')
    ears_train = ears(face_train)
    ears_eval = ears(face_eval)
    whiskers = layers.Dense(units=1,name='whiskers')
    whiskers_train = whiskers(face_train)
    whiskers_eval = whiskers(face_eval)
    # Body hierarchy
    body = layers.Dense(units=2,name='body')
    body_train = body(longest_vector_train)
    body_eval = body(longest_vector_eval)
    paws = layers.Dense(units=1,name='paws')
    paws_train = paws(body_train)
    paws_eval = paws(body_eval)
    tail = layers.Dense(units=1,name='tail')
    tail_train = tail(body_train)
    tail_eval = tail(body_eval)
    # Colour hierarchy
    colour = layers.Dense(units=4,name='colour')
    colour_train = colour(longest_vector_train)
    colour_eval = colour(longest_vector_eval)
    brown = layers.Dense(units=1,name='brown')
    brown_train = brown(colour_train)
    brown_eval = brown(colour_eval)
    black = layers.Dense(units=1,name='black')
    black_train = black(colour_train)
    black_eval = black(colour_eval)
    grey = layers.Dense(units=1,name='grey')
    grey_train = grey(colour_train)
    grey_eval = grey(colour_eval)
    white = layers.Dense(units=1,name='white')
    white_train = white(colour_train)
    white_eval = white(colour_eval)

    # Now, build the model
    hierarchy_train_model = models.Model([x, y], [out_caps,face_train,eyes_train,mouth_open_train,snout_train,ears_train,whiskers_train,body_train,paws_train,tail_train,colour_train,brown_train,black_train,grey_train,white_train])
    hierarchy_eval_model = models.Model(x, [out_caps,face_eval,eyes_eval,mouth_open_eval,snout_eval,ears_eval,whiskers_eval,body_eval,paws_eval,tail_eval,colour_eval,brown_eval,black_eval,grey_eval,white_eval])
    #------------------------------------------------------------------------------------------------------------------------------

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([z, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(z)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(256, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(512, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([z, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model, hierarchy_train_model, hierarchy_eval_model


def total_loss(y_true, y_pred):
    """
    Total loss = Margin loss for Eq.(4) in Hinton et. al. + KL Divergence loss of the VAE. When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    Also adding the KL Divergence loss of the distributions. If training is poor, try to reconstruct this loss.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """

    # y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    # y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32) 
     
    # L is of dimension(?,10)  

    # Changes made here to fix the KLD Loss term error
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    '''
    # kl_loss is of dimension (?,10,16). Be careful when you add the two. This simplification of KLD is for Gaussian only
    kl_loss = tf.convert_to_tensor(log_variance, dtype=tf.float32) + tf.convert_to_tensor(log_variance, dtype=tf.float32) - tf.cast(K.square(tf.convert_to_tensor(mean, dtype=tf.float32)), tf.float32) - tf.cast(K.exp(tf.convert_to_tensor(log_variance, dtype=tf.float32)), tf.float32)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    L+= kl_loss
    return tf.convert_to_tensor(K.mean(K.sum(L, axis=1)), dtype=tf.float32)
    '''
    kl_loss = tf.convert_to_tensor(losses.kullback_leibler_divergence(y_true, y_pred))
    return tf.convert_to_tensor(K.mean(K.sum(L,1)+kl_loss))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[total_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    
    # Training without data augmentation:

    model.fit([x_train, y_train], [y_train, x_train],
     batch_size=args.batch_size, 
     epochs=args.epochs,
     validation_data=[[x_test, y_test], [y_test, x_test]], 
     callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#
	"""
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("X Train:", x_train)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def get_file_name(path):
    head, tail = ntpath.split(path)
    return str(tail) or str(ntpath.basename(head))

def load_custom_dataset(dataset_path):
    '''
    Function to use custom dataset
    '''
    
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    classes = ['cats', 'dogs', 'foxes', 'hyenas', 'wolves']
    class_dict = {'cats':0, 'dogs':1, 'foxes':2, 'hyenas':3, 'wolves':4}

    y_train_dataframe = pd.read_csv(dataset_path+'animals.csv')

    append_count = 0
    for class_name in classes:
        print("Processing class", class_name+"..")
        img_dir = dataset_path+str(class_name)+'/'
        data_path = os.path.join(img_dir,'*g')
        files = glob.glob(data_path)
        for current_file in files:
            random_number = random.randint(1,10)

            if(random_number == 7):
                img = cv2.imread(current_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (28, 28))
                x_test.append(img)
                y_test.append(class_dict[class_name])

            # for index, row in y_train_dataframe.iterrows():
            #     if get_file_name(current_file) == row['File Name']:
            #         y_train.append(row['Features'])
            else:
                img = cv2.imread(current_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (28, 28))
                x_train.append(img)
                append_count += 1
                y_train.append(class_dict[class_name])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_test = to_categorical(y_test.astype('float32'))

    # Uncomment to debug
    # print("Length of training set:", len(x_train), "labels:", len(y_train))
    # print("Length of training set:", len(x_test), "labels:", len(y_test))

    # RETURN HERE
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--dataset', default=dataset_path, type=str, help="Relative path to the custom dataset to use")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    # (x_train, y_train), (x_test, y_test) = load_mnist()

    # Testing custom data reader
    (x_train, y_train), (x_test, y_test) = load_custom_dataset(args.dataset)

    # define model
    model, eval_model, manipulate_model, hierarchy_train_model, hierarchy_eval_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()
    hierarchy_train_model.summary()
    hierarchy_eval_model.summary()
    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)
