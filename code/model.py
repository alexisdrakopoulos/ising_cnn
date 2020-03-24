from sys import argv
import os
from datetime import datetime
from os.path import isfile
from warnings import warn
from configparser import ConfigParser, ExtendedInterpolation
from utils import create_directories, experimental_log, convert_to_categories

# assign a single GPU if so chosen by user in args
try:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # assign GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = argv[1]
except IndexError:
    pass

# Keras/machine learning imports
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras import losses
from keras import metrics
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback

# Read config file for paths
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("code/config.ini")


def load_data(model_type):
    """
    Imports data from numpy files saved to disk
    """

    data = np.load(f"{config['Paths']['ising data']}/ising_data.npz")

    return data["training_data"], data["training_labels"], data["validation_data"], data["validation_labels"]


def cnn_regressor(model_iteration=1,
                  model_type="regression",
                  activation="relu",
                  optimizer="adam",
                  dropout=False,
                  batchnorm=False,
                  batchnorm_order="before",
                  save_model=True,
                  batch_size=128,
                  epochs=40,
                  learning_rate=0.00002,
                  training_data_length=200_000,
                  shallow_network=False,
                  predictions=False,
                  logging_loss=True):
    """
    CNN Regression model using vggnet16 and dropout layers

    Inputs:
        model_iteration - The experiment run to keep track of models
        model_type - regression or classification
        activation - relu or lrelu depending on activation to use in model
        dropout - Bool on whether to use dropout in between each set of layers
        batchnorm - Bool on whether to use batchnorm in between each set of layers
        batchnorm_order - before or after activations
        save_model - Bool on whether to save final model to disk, best model is saved
        batch_size - The batch size to use
        epochs - The number of epochs to run through, auto-stop procedure exists
        predictions - Bool whether to run the predictions

    Outputs:
        best model is saved to disk, and if save_model is True final model also saved
        a csv file containing training data is also saved to disk.
    """

    # Check args
    if batchnorm_order not in ("before", "after"):
        raise ValueError("batch_norm arg needs to be 'before' or 'after'.")

    if model_type not in ("regression", "classification"):
        raise ValueError("model_type arg needs to be 'regression' or 'classification'")

    if activation not in ("relu", "lrelu"):
        raise ValueError("Sorry at the moment only relu and lrelu is implemented.")

    # Convert args back to bools and ints from argparse command line
    model_iteration = int(model_iteration)
    dropout = float(dropout)
    batchnorm = eval(batchnorm.capitalize())
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    training_data_length = int(training_data_length)
    shallow_network = eval(shallow_network.capitalize())

    # Logging data to txt file
    experimental_log(model_iteration,
                     model_type,
                     activation,
                     optimizer,
                     dropout,
                     batchnorm,
                     batchnorm_order,
                     batch_size,
                     epochs,
                     learning_rate,
                     training_data_length,
                     shallow_network)

    # Setting file names
    if model_type == "regression":
        add_type = "reg"
    elif model_type == "classification":
        add_type = "clas"

    training_file = f"{config['Paths']['training logs']}/training_" + str(model_iteration) + ".csv"
    final_model = f"{config['Paths']['models']}/fmodel_" + str(model_iteration) + "_" + add_type + ".h5"
    best_model = f"{config['Paths']['models']}/bmodel_" + str(model_iteration) + "_" + add_type + ".h5"

    # input image dimensions
    img_rows, img_cols = 128, 128
    input_shape = (img_rows, img_cols, 1)

    # Import the data
    print("Starting Model")
    print("Importing Data and Labels")
    training_data, training_labels, val_data, val_labels = load_data(model_type)

    print(f"Using {training_data_length} training samples")
    training_data = training_data[:training_data_length]
    training_labels = training_labels[:training_data_length]
    val_data = val_data[:training_data_length//5]
    val_labels = val_labels[:training_data_length//5]

    # Reshape the data for channels last
    training_data = training_data.reshape(training_data.shape[0], img_rows, img_cols, 1)
    val_data = val_data.reshape(val_data.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)


    # Encode labels if classification is being used
    if model_type == "classification":
        num_classes = 10
        val_labels = convert_to_categories(val_labels, num_classes)
        training_labels = convert_to_categories(training_labels, num_classes)
        val_labels = to_categorical(val_labels.astype(int), num_classes)
        training_labels = to_categorical(training_labels.astype(int), num_classes)


    def optimization_algorithm(optimizer):
        
        if optimizer == "adam":
            return Adam(lr=learning_rate)
        elif optimizer == "sgd":
            return SGD(lr=learning_rate, momentum=0.9, nesterov=True)


    def activation_function(activation):
        """
        custom build activation function to return relu or lrelu

        Inputs:
            activation - string, either relu or lrelu
        Outputs:
            returns an activation to add to keras model
        """

        if activation == "relu":
            return Activation("relu")

        elif activation == "lrelu":
            return LeakyReLU()


    def final_activation(model,
                         pooling=True,
                         dropout=0,
                         batchnorm=False,
                         batchnorm_order="before"):
        """
        To shorten the conditionals required for the final activation
        at the end of each set of convolutional operations.

        Inputs:
            model - the keras model to add the layers to
            pooling - bool as to whether to pool, set False if dense
        Outputs:
            adds layers such as batchnormalization, pooling and dropout
        """

        if batchnorm and batchnorm_order == "before":
            model.add(BatchNormalization(momentum=0.9, epsilon=0.1))

        model.add(activation_function(activation=activation))

        if batchnorm and batchnorm_order == "after":
            model.add(BatchNormalization(momentum=0.9, epsilon=0.1))

        if pooling:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        if dropout > 0:
            model.add(Dropout(dropout))


    print("Running CNN")
    print('Training data shape:', training_data.shape)
    print(training_data.shape[0], 'training samples')
    print(val_data.shape[0], 'test samples')


    def conv_block(layers, neurons, first_layer=False, activation=activation, pooling=True):
        """
        Function to add Conv2D layers to a model

        Inputs:
            layers - int specifying number of convolutional layers to add
            neurons - int specifying number of neurons for each conv layer to have
            input_shape - bool specifying whether to pass input_shape to first layer
            activation - activation to use
            pooling - bool specifying whether model uses maxpool or downsampling
        """

        # Specifying input shape if it's the first conv block
        if first_layer:
            model.add(Conv2D(neurons, (3, 3), input_shape=input_shape, padding="same"))
            model.add(activation_function(activation=activation))

            for _ in range(layers-2):
                model.add(Conv2D(neurons, (3, 3), padding="same"))
                model.add(activation_function(activation=activation))

        else:
            for _ in range(layers-1):
                model.add(Conv2D(neurons, (3, 3), padding="same"))
                model.add(activation_function(activation=activation))

        # Remove bias if using batchnorm
        if batchnorm:
            bias = False
        else:
            bias = True

        # Pooling otherwise downsample by having stirde of 2 as final layer
        if pooling:
            model.add(Conv2D(neurons, (3, 3), padding="same", use_bias=bias))
            model.add(activation_function(activation=activation))
            final_activation(model=model, pooling=True,
                             dropout=dropout, batchnorm=batchnorm,
                             batchnorm_order=batchnorm_order)
        else:
            model.add(Conv2D(neurons, (3, 3), 2, padding="same", use_bias=bias))
            model.add(activation_function(activation=activation))
            final_activation(model=model, pooling=False,
                             dropout=dropout, batchnorm=batchnorm,
                             batchnorm_order=batchnorm_order)


    def fc_block(layers, neurons, flatten=False, add_activation=True):
        """
        Function to add fully connected layers to the model

        Inputs:
            layers - int specifying number of fully connected layers
            neurons - int specifying number of neurons per layer
            flatten - bool argument to pass to flatten input before passing to layers
            final_activation - bool whether to apply final_activation func
        """

        if flatten:
            model.add(Flatten())

        # Remove bias if using batchnorm
        if batchnorm:
            bias = False
        else:
            bias = True

        for _ in range(layers):
            model.add(Dense(neurons, use_bias=bias))
            if add_activation:
                final_activation(model=model, pooling=False,
                                 dropout=dropout, batchnorm=False,
                                 batchnorm_order=batchnorm_order)


    model = Sequential()
    conv_block(layers=2, neurons=64, pooling=True, first_layer=True)
    conv_block(layers=2, neurons=128, pooling=True)
    conv_block(layers=3, neurons=256, pooling=True)
    conv_block(layers=3, neurons=512, pooling=True)
    if not shallow_network:
        conv_block(layers=3, neurons=512, pooling=True)
    fc_block(layers=2, neurons=4096, flatten=True)

    if model_type == "regression":
        fc_block(layers=1, neurons=1000, add_activation=False)
        model.add(Dense(1))

    elif model_type == "classification":
        fc_block(layers=1, neurons=1000, add_activation=False)
        final_activation(model=model, pooling=False,
                         dropout=False, batchnorm=False)
        model.add(Dense(num_classes, activation="softmax"))

    optimizer = optimization_algorithm(optimizer)

    # Losses and compiler
    if model_type == "regression":
        loss = losses.mean_squared_error
        metric = ["mse", tf.keras.losses.mape]
    elif model_type == "classification":
        loss = losses.categorical_crossentropy
        metric = ["categorical_accuracy"]

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metric)

    # Call Backs for early stopping, logging, checkpoints and lr reduction
    csv_logger = CSVLogger(training_file)
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.0002,
                               patience=12,
                               verbose=1,
                               mode='min')
    mcp_save = ModelCheckpoint(best_model,
                               save_best_only=True,
                               monitor='val_loss',
                               mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.25,
                                       patience=4,
                                       verbose=1,
                                       min_delta=0.0002,
                                       mode='min',
                                       min_lr=1e-7)

    # Custom callback for per-batch metrics and loss
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            if model_type == "classification":
                self.accuracy = []
            elif model_type == "regression":
                self.mape = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            if model_type == "classification":
                self.accuracy.append(logs.get('acc'))
            elif model_type == "regression":
                self.mape.append(logs.get("mean_absolute_percentage_error"))

    loss_history = LossHistory()

    # callbacks for the keras model
    callbacks = [loss_history,
                 early_stop,
                 mcp_save,
                 reduce_lr_loss,
                 csv_logger]

    history = model.fit(training_data, training_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(val_data, val_labels),
                        callbacks=callbacks)

    if logging_loss:
        print("Saving per batch losses")
        if model_type == "classification":
            loss_histories = np.vstack((np.array(loss_history.losses),
                                        np.array(loss_history.accuracy)))
        elif model_type == "regression":
            loss_histories = np.vstack((np.array(loss_history.losses),
                                        np.array(loss_history.mape)))

        loss_histories_name = f"{config['Paths']['loss histories']}/" + str(model_iteration) + "loss_histories"
        np.save(loss_histories_name, loss_histories)


    print("Running evaluation on validation data")
    score = model.evaluate(val_data, val_labels, verbose=0)
    print(score)

    if save_model:
        print("Saving model")
        model.save(final_model)

    if predictions:
        print("Running predictions on test data")
        test_data = np.load(f"{config['Paths']['ising data']}/test_data.npz")
        test_data = test_data["data"]
        predictions = model.predict(test_data)
        prediction_name = config['Paths']['predictions'] + "/predictions_" + str(model_iteration) + "_" + str(model_type) + ".npy"
        np.save(prediction_name, predictions)


print("Training Network")
create_directories()
predictions = cnn_regressor(model_iteration=argv[2],
                            model_type=argv[3],
                            activation=argv[4],
                            optimizer=argv[5],
                            dropout=argv[6],
                            batchnorm=argv[7],
                            batchnorm_order=argv[8],
                            batch_size=argv[9],
                            learning_rate=argv[10],
                            training_data_length=argv[11],
                            shallow_network=argv[12])
