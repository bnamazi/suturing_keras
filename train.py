import os
import glob
import keras
from DataGenerator import VideoFrameGenerator
from flow import OpticalFlowGenerator
from models import cnn, cnn_rnn, conv_lstm, CNN3D, Attention
import utils
import tensorflow
from keras.layers import (
    Conv2D,
    BatchNormalization,
    Reshape,
    TimeDistributed,
    Flatten,
    MaxPool2D,
    GlobalMaxPool2D,
    Input,
    Dense,
    Dropout,
    Concatenate,
    Lambda,
    GlobalAvgPool2D,
    MaxPooling2D,
    add,
    GlobalAveragePooling3D,
    GlobalAveragePooling2D,
)
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3

# import keras.backend as K
import keras as K
from keras.models import Sequential, Model
import wandb
from wandb.keras import WandbCallback

wandb.init(project="suturing_keras", entity="bnamazi")

config = wandb.config

config["WANDB"] = True

# use sub directories names as classes
# classes = [i.split(os.path.sep)[1] for i in glob.glob('data/*')]
# classes = ["pass", "fail"]
classes = [
    "Cutting Tails",
    "Knot 1 lay",
    "Knot 2 lay",
    "Needle and suture withdrawal",
    "Needle Loading and insertion",
    "Single throw 1",
    "Single throw 2",
    "Surgeons knot lay",
    "Surgeons knot wrap",
]
classes.sort()
# class_weight = {0:1, 1:1}

# some global params
h = 224
w = 224
SIZE = (w, h)
CHANNELS = 3
config.NBFRAME = 16
config.BS = 32
EPOCHS = 100
config.learning_rate = 0.005


for k in range(1, 11):
    path = f"./data/steps/"
    glob_pattern_train = str(path) + "train/{classname}/*.*"
    glob_pattern_test = str(path) + "test/{classname}/*.*"

    checkpoint_filepath = f"./ckp/steps.h5py"

    # for data augmentation
    data_aug_train = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.05,
        horizontal_flip=False,
        vertical_flip=False,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # brightness_range=[0.8,1.2],
        rotation_range=0,
    )
    data_aug_test = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        width_shift_range=0.0,
        height_shift_range=0.0,
    )
    # Create video frame generator
    train = VideoFrameGenerator(
        classes=classes,
        glob_pattern=glob_pattern_train,
        nb_frames=config.NBFRAME,
        shuffle=True,
        batch_size=config.BS,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug_train,
        use_frame_cache=True,
        train=True,
    )

    valid = VideoFrameGenerator(
        classes=classes,
        glob_pattern=glob_pattern_test,
        nb_frames=config.NBFRAME,
        shuffle=False,
        batch_size=1,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        transformation=data_aug_test,
        use_frame_cache=True,
        train=False,
    )

    # utils.show_sample(train)

    # def build_convnet(shape=(NBFRAME, 112, 112, CHANNELS)):
    shape = (config.NBFRAME, h, w, CHANNELS)
    # model = keras.Sequential()
    input = Input(shape)

    # output = cnn_rnn(input, shape)
    x3d = CNN3D(input, shape)
    model2 = K.Model(inputs=input, outputs=x3d.call(input), name="X3D")
    # model.load_weights('./checkpoints/model')
    checkpoint = tf.train.Checkpoint(model2)
    checkpoint.restore("./ckp/model").expect_partial()
    # model2.layers.pop()
    # model2.layers.pop()
    # new_model = Sequential()
    # new_model.add(model)
    model2.summary()
    new_fc = K.layers.Dense(
        units=9,
        use_bias=True,
        name="fc_2_new",
        kernel_regularizer=K.regularizers.L2(5e-5)
        # activation='softmax'
    )
    soft_max = K.layers.Softmax(dtype="float32")

    model = Sequential()
    for layer in model2.layers[:-3]:  # go through until last layer
        model.add(layer)
    # model.summary()
    model.add(Flatten())
    model.add(new_fc)
    model.add(soft_max)
    # new_.add(K.layers.Softmax(dtype='float32'))
    # out = soft_max(new_fc(model2.layers[-4].output))
    # out = tf.reshape(out, (-1, 9))
    # model = K.Model(inputs=model2.input, outputs=out)
    # for layer in model.layers:
    #     print(layer)
    #     layer.trainable = True
    # del model2
    model.build((None, config.NBFRAME, h, w, CHANNELS))
    # model = tensorflow.keras.Model(inputs=input, outputs=output)
    model.summary()
    # visualkeras.layered_view(model, to_file='block_diagram.png', legend=True)#, spacing=10, scale_xy=1, scale_z=0.7, max_z=30).show()
    optimizer = keras.optimizers.SGD(config.learning_rate, momentum=0.9)

    # if os.path.exists(checkpoint_filepath):
    #     model.load_weights(checkpoint_filepath)
    model.compile(optimizer, "categorical_crossentropy", metrics=["acc"])

    # create a "chkp" directory before to run that
    # because ModelCheckpoint will write models inside
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(verbose=1, factor=0.5, patience=15),
        keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_acc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_acc",
            min_delta=0,
            patience=50,
            verbose=1,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        ),
        WandbCallback(  # training_data=train,
            # log_gradients=True,
            log_weights=True
        ),
    ]
    model.fit(
        train,
        validation_data=valid,
        verbose=1,
        epochs=EPOCHS,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=8
        # class_weight=class_weight
    )

    model.load_weights(checkpoint_filepath)

    test_scores = model.evaluate(valid, verbose=2)
    print(test_scores)
    with open("results.txt", "a") as f:
        f.write(f"{k} - {test_scores}\n")

    predictions = model.predict(valid)

    for cls in classes:
        for index, file in enumerate(
            glob.glob(glob_pattern_test.format(classname=cls))
        ):
            with open("results.txt", "a") as f:
                f.write(f"{file}, {predictions[index]}\n")

    K.backend.clear_session()
    del model
