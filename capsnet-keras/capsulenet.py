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
    About 110 seconds per epoch on a single Nvidia GTX 1070 GPU card
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
dataset_path = "../../Dataset/Animals/"

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
    #----------------------------------------------------------------------------------------------------------------------------
    
    y = layers.Input(shape=((n_class),))
    masked_by_y = Mask()([z, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(z)  # Mask using the capsule with maximal length. For prediction

    longest_vector_train = masked_by_y
    longest_vector_eval = masked

    # Keep adding hierarchies

    # Face hierarchy
    face = layers.Dense(units=9,activation='relu',name='face')
    face_train = face(longest_vector_train)
    face_eval = face(longest_vector_eval)
    face_output = layers.Dense(units=1,activation='softmax',name='face_output')
    face_output_train = face_output(face_train)
    face_output_eval = face_output(face_eval)
    eyes = layers.Dense(units=1,activation='relu',name='eyes')
    eyes_train = eyes(face_train)
    eyes_eval = eyes(face_eval)
    mouth = layers.Dense(units=1,activation='relu',name='mouth')
    mouth_train = mouth(face_train)
    mouth_eval = mouth(face_eval)
    snout = layers.Dense(units=1,activation='relu',name='snout')
    snout_train = snout(face_train)
    snout_eval = snout(face_eval)
    ears = layers.Dense(units=1,activation='relu',name='ears')
    ears_train = ears(face_train)
    ears_eval = ears(face_eval)
    whiskers = layers.Dense(units=1,activation='relu',name='whiskers')
    whiskers_train = whiskers(face_train)
    whiskers_eval = whiskers(face_eval)
    nose = layers.Dense(units=1,activation='relu',name='nose') # NEW
    nose_train = nose(face_train)
    nose_eval = nose(face_eval)
    teeth = layers.Dense(units=1,activation='relu',name='teeth') # NEW
    teeth_train = teeth(face_train)
    teeth_eval = teeth(face_eval)
    beak = layers.Dense(units=1,activation='relu',name='beak') # NEW
    beak_train = beak(face_train)
    beak_eval = beak(face_eval)
    tongue = layers.Dense(units=1,activation='relu',name='tongue') # NEW
    tongue_train = tongue(face_train)
    tongue_eval = tongue(face_eval)
    # Body hierarchy
    body = layers.Dense(units=12,activation='relu',name='body')
    body_train = body(longest_vector_train)
    body_eval = body(longest_vector_eval)
    body_output = layers.Dense(units=1,activation='softmax',name='body_output')
    body_output_train = body_output(body_train)
    body_output_eval = body_output(body_eval)
    wings = layers.Dense(units=1,activation='relu',name='wings') # NEW
    wings_train = wings(body_train)
    wings_eval = wings(body_eval)
    paws = layers.Dense(units=1,activation='relu',name='paws')
    paws_train = paws(body_train)
    paws_eval = paws(body_eval)
    tail = layers.Dense(units=1,activation='relu',name='tail')
    tail_train = tail(body_train)
    tail_eval = tail(body_eval)
    legs = layers.Dense(units=1,activation='relu',name='legs') # NEW
    legs_train = legs(body_train)
    legs_eval = legs(body_eval)
    surface = layers.Dense(units=1,activation='relu',name='surface') # NEW
    surface_train = surface(body_train)
    surface_eval = surface(body_eval)
    arm_rest = layers.Dense(units=1,activation='relu',name='arm_rest') # NEW
    arm_rest_train = arm_rest(body_train)
    arm_rest_eval = arm_rest(body_eval)
    base = layers.Dense(units=1,activation='relu',name='base') # NEW
    base_train = base(body_train)
    base_eval = base(body_eval)
    pillows = layers.Dense(units=1,activation='relu',name='pillows') # NEW
    pillows_train = pillows(body_train)
    pillows_eval = pillows(body_eval)
    cushions = layers.Dense(units=1,activation='relu',name='cushions') # NEW
    cushions_train = cushions(body_train)
    cushions_eval = cushions(body_eval)
    drawer = layers.Dense(units=1,activation='relu',name='drawer') # NEW
    drawer_train = drawer(body_train)
    drawer_eval = drawer(body_eval)
    knob = layers.Dense(units=1,activation='relu',name='knob') # NEW
    knob_train = knob(body_train)
    knob_eval = knob(body_eval)
    mattress = layers.Dense(units=1,activation='relu',name='mattress') # NEW
    mattress_train = mattress(body_train)
    mattress_eval = mattress(body_eval)
    # Colour hierarchy
    colour = layers.Dense(units=8,activation='relu',name='colour')
    colour_train = colour(longest_vector_train)
    colour_eval = colour(longest_vector_eval)
    colour_output = layers.Dense(units=1,activation='softmax',name='colour_output')
    colour_output_train = colour_output(colour_train)
    colour_output_eval = colour_output(colour_eval)
    brown = layers.Dense(units=1,activation='relu',name='brown')
    brown_train = brown(colour_train)
    brown_eval = brown(colour_eval)
    black = layers.Dense(units=1,activation='relu',name='black')
    black_train = black(colour_train)
    black_eval = black(colour_eval)
    grey = layers.Dense(units=1,activation='relu',name='grey')
    grey_train = grey(colour_train)
    grey_eval = grey(colour_eval)
    white = layers.Dense(units=1,activation='relu',name='white')
    white_train = white(colour_train)
    white_eval = white(colour_eval)
    purple = layers.Dense(units=1,activation='relu',name='purple') # NEW
    purple_train = purple(colour_train)
    purple_eval = purple(colour_eval)
    pink = layers.Dense(units=1,activation='relu',name='pink') # NEW
    pink_train = pink(colour_train)
    pink_eval = pink(colour_eval)
    yellow = layers.Dense(units=1,activation='relu',name='yellow') # NEW
    yellow_train = yellow(colour_train)
    yellow_eval = yellow(colour_eval)
    turqoise = layers.Dense(units=1,activation='relu',name='turqoise') # NEW
    turqoise_train = turqoise(colour_train)
    turqoise_eval = turqoise(colour_eval)
    # Alternate / Unknown hierarchy
    unknown = layers.Dense(units=1,activation='relu',name='unknown') # NEW
    unknown_train = unknown(longest_vector_train)
    unknown_eval = unknown(longest_vector_eval)

    # Now, build both the models
    hierarchy_train_model = models.Model([x, y], [out_caps,face_output_train,eyes_train,mouth_train,snout_train,ears_train,whiskers_train,nose_train,teeth_train,beak_train,tongue_train,body_output_train,wings_train,paws_train,tail_train,legs_train,surface_train,arm_rest_train,base_train,pillows_train,cushions_train,drawer_train,knob_train,mattress_train,colour_output_train,brown_train,black_train,grey_train,white_train,purple_train,pink_train,yellow_train,turqoise_train,unknown_train])
    hierarchy_eval_model  = models.Model(x,      [out_caps,face_output_eval,eyes_eval,mouth_eval,snout_eval,ears_eval,whiskers_eval,nose_eval,teeth_eval,beak_eval,tongue_eval,body_output_eval,wings_eval,paws_eval,tail_eval,legs_eval,surface_eval,arm_rest_eval,base_eval,pillows_eval,cushions_eval,drawer_eval,knob_eval,mattress_eval,colour_output_eval,brown_eval,black_eval,grey_eval,white_eval,purple_eval,pink_eval,yellow_eval,turqoise_eval,unknown_eval])
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
     
    # L is of dimension(?,5)  

    # Changes made here to fix the KL Divergence loss term error
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    '''
    # kl_loss is of dimension (?,5,16). Be careful when you add the two. This simplification of KLD is for Gaussian only
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
    # y_train_output and y_test_output are the list of words
    (x_train, y_train, y_train_output), (x_test, y_test, y_test_output) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # Compile the model

    # all_losses - 34 is the number of outputs
    all_losses = ['mse' for _ in range(34)]
    # change first one to total_loss
    all_losses[0] = total_loss

    # all_loss_weights - 34 is the number of outputs
    # We're giving a weight of 0.5 for the final outputs to the total weight
    all_loss_weights = [0.5 for _ in range(34)]
    all_loss_weights[0] = 1.

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=all_losses,
                  loss_weights=all_loss_weights,
                  metrics={'capsnet': 'accuracy'})

    
    # Training without data augmentation (preferred) :
    # Combine y_train and y_train_output to get an uniform vector
    y_train_list=[0]    # Dummy value to avoid index out of bounds in next line
    y_train_list[0] = y_train

    # To reshape  y_train_output from [None,33] to [33,None]
    y_train_output = np.array([np.array(_) for _ in zip(*y_train_output)])
    for output in y_train_output:
        y_train_list.append(output)
    
    # Combine y_test and y_test_output to get an uniform vector
    y_test_list=[0]    # Dummy value to avoid index out of bounds in next line
    y_test_list[0] = y_test

    # To reshape  y_test_output from [None,33] to [33,None]
    y_test_output = np.array([np.array(_) for _ in zip(*y_test_output)])
    for output in y_test_output:
        y_test_list.append(output)

    model.fit([x_train, y_train], y_train_list,
     batch_size=args.batch_size, 
     epochs=args.epochs,
     validation_data=[[x_test, y_test], y_test_list], 
     callbacks=[log, tb, checkpoint, lr_decay])
    
    '''
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
                        validation_data=[[x_test, y_test], [y_test, y_test_output]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#
    '''

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test, y_test_output = data
    y_pred, y_test_output_pred = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    # Test accuracy for the predicted output words
    # To be done


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

def build_output(features):
    # Builds the output according to the output format in the hierarchy_( train | eval )_model
    # Length of output vector
    output = [0 for _ in range(33)]
    # -------- Volatile (under processing) ------------------------- <BEGIN>
    nothing_present_flag = True
    # Order - face,eyes,mouth,snout,ears,whiskers,nose,teeth,beak,tongue,body,wings,paws,tail,legs,surface,arm_rest,base,pillows,cushions,drawer,knob,mattress,colour,brown,black,grey,white,purple,pink,yellow,turqoise,unknown
    if 'face' in features:
    	output[0]=1
    	nothing_present_flag = False
    if 'eyes' in features:
    	output[1]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'mouth' in features:
    	output[2]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'snout' in features:
    	output[3]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'ears' in features:
    	output[4]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'whiskers' in features:
    	output[5]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'nose' in features:
    	output[6]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'teeth' in features:
    	output[7]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'beak' in features:
    	output[8]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'tongue' in features:
    	output[9]=1
    	output[0]=1
    	nothing_present_flag = False
    if 'body' in features:
    	output[10]=1
    	nothing_present_flag = False
    if 'wings' in features:
    	output[11]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'paws' in features:
    	output[12]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'tail' in features:
    	output[13]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'legs' in features:
    	output[14]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'surface' in features:
    	output[15]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'arm rests' in features:
    	output[16]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'base' in features:
    	output[17]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'pillows' in features:
    	output[18]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'cushions' in features:
    	output[19]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'drawers' in features:
    	output[20]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'knobs' in features:
    	output[21]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'mattress' in features:
    	output[22]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'colour' in features:
    	output[23]=1
    	nothing_present_flag = False
    if 'brown' in features:
    	output[24]=1
    	output[23]=1
    	nothing_present_flag = False
    if 'black' in features:
    	output[25]=1
    	output[23]=1
    	nothing_present_flag = False
    if 'grey' in features:
    	output[26]=1
    	output[23]=1
    	nothing_present_flag = False
    if 'white' in features:
    	output[27]=1
    	output[23]=1
    	nothing_present_flag = False
    if 'purple' in features:
    	output[28]=1
    	output[23]=1
    	nothing_present_flag = False
    if 'pink' in features:
    	output[29]=1
    	output[23]=1
    	nothing_present_flag = False
    if 'yellow' in features:
    	output[30]=1
    	output[23]=1
    	nothing_present_flag = False
    if 'turqoise' in features:
    	output[31]=1
    	output[23]=1
    	nothing_present_flag = False
    # Other "similar" cases
    if 'eye' in features:
    	output[1]=0.5
    	output[0]=1
    	nothing_present_flag = False
    if 'ear' in features:
    	output[4]=0.5
    	output[0]=1
    	nothing_present_flag = False
    if 'wing' in features:
    	output[11]=0.5
    	output[10]=1
    	nothing_present_flag = False
    if 'paw' in features:
    	output[12]=0.5
    	output[10]=1
    	nothing_present_flag = False
    if 'leg' in features:
    	output[14]=0.5
    	output[10]=1
    	nothing_present_flag = False
    if 'rectangular surface' in features:
    	output[15]=1
    	output[10]=1
    	nothing_present_flag = False
    if 'circular surface' in features:
    	output[15]=2
    	output[10]=1
    	nothing_present_flag = False
    if 'arm rest' in features:
    	output[16]=0.5
    	output[10]=1
    	nothing_present_flag = False
    if 'pillow' in features:
    	output[18]=0.5
    	output[10]=1
    	nothing_present_flag = False
    if 'cushion' in features:
    	output[19]=0.5
    	output[10]=1
    	nothing_present_flag = False
    if 'drawer' in features:
        output[20]=0.5
        output[10]=1
        nothing_present_flag = False
    if 'knob' in features:
        output[21]=0.5
        output[10]=1
        nothing_present_flag = False
    if 'silver' in features:
    	output[27]=0.5
    	output[23]=1
    	nothing_present_flag = False
    if 'transparent' in features:
    	output[27]=0
    	output[23]=1
    	nothing_present_flag = False
    if 'golden' in features:
    	output[30]=0.5
    	output[23]=1
    	nothing_present_flag = False
    
    # For 'unknown' case
    if nothing_present_flag:
        output[:-1] = [0]*(len(output)-1)
        output[-1]=1
    output = np.array(output)
    # -------- Volatile (under processing) ------------------------- <END>
    return output

def load_custom_dataset(dataset_path):

    # Function to use custom dataset
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    y_train_output = []
    y_test_output = []

    classes = {'animals':['cats', 'dogs', 'foxes', 'hyenas', 'wolves'],'birds':['ducks','eagles','hawks','parrots','sparrows'],'furniture':['chair','sofa','table']}
    class_encodings = {'cats':0, 'dogs':1, 'foxes':2, 'hyenas':3, 'wolves':4, 'ducks':5, 'eagles':6, 'hawks':7, 'parrots':8, 'sparrows':9, 'chair':10, 'sofa':11, 'table':12}
    # classes = {'animals':['cats', 'dogs', 'foxes', 'hyenas', 'wolves'],'birds':['ducks','eagles','parrots','sparrows'],'furniture':['chair','sofa','table']}
    # class_encodings = {'cats':0, 'dogs':1, 'foxes':2, 'hyenas':3, 'wolves':4, 'ducks':5, 'eagles':6, 'parrots':8, 'sparrows':9, 'chair':10, 'sofa':11, 'table':12}

    for class_ in classes:
        dataset_path = "../Dataset/"+class_[0].upper()+class_[1:]+'/'

        y_train_dataframe = pd.read_csv("./csv_folder/"+class_+'.csv', encoding = "ISO-8859-1")
        for sub_class in classes[class_]:
            print("Processing class", sub_class+"..")
            img_dir = dataset_path+str(sub_class)+'/'
            data_path = os.path.join(img_dir,'*g')
            files = glob.glob(data_path)

            for current_file in files:
                random_number = random.randint(1,10)

                if(random_number == 7):
                    img = cv2.imread(current_file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (28, 28))
                    
                    # y_test_output logic with x_test and y_test append
                    for index, row in y_train_dataframe.iterrows():
                        if get_file_name(current_file) == row['File Name']:
                            x_test.append(img)
                            # if(sub_class == 'table'):
                            y_test.append(class_encodings[sub_class])
                            y_test_features = row['Features']
                            y_test_output.append(build_output(y_test_features))
                            break
                else:
                    img = cv2.imread(current_file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (28, 28))
                    
                    # y_train_output logic with x_train and y_train append
                    for index, row in y_train_dataframe.iterrows():
                        if get_file_name(current_file) == row['File Name']:
                            x_train.append(img)
                            y_train.append(class_encodings[sub_class])
                            
                            y_train_features = row['Features']
                            y_train_output.append(build_output(y_train_features))
                            break

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train_output = np.array(y_train_output)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test_output = np.array(y_test_output)
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train, y_train_output), (x_test, y_test, y_test_output)

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

    # Load data
    (x_train, y_train, y_train_output), (x_test, y_test, y_test_output) = load_custom_dataset(args.dataset)

    # Define model
    model, eval_model, manipulate_model, hierarchy_train_model, hierarchy_eval_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    #model.summary()
    #hierarchy_train_model.summary()
    #hierarchy_eval_model.summary()
    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        # Send hierarchy_train_model along with changes in data format
        train(model=hierarchy_train_model, data=((x_train, y_train, y_train_output), (x_test, y_test, y_test_output)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        # manipulate_latent(manipulate_model, (x_test, y_test), args)
        # Send hierarchy_eval_model along with changes in data format
        test(model=hierarchy_eval_model, data=(x_test, y_test, y_test_output), args=args)
