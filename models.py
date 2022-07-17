from keras.layers import (
    Conv2D,
    BatchNormalization,
    Flatten,
    LSTM,
    Bidirectional,
    ConvLSTM2D,
    GRU,
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
    GlobalAveragePooling2D,
    TimeDistributed,
    Conv3D,
    GlobalAveragePooling3D,
    MaxPool3D,
    GlobalAveragePooling1D,
    Reshape,
    Softmax,
)
from keras.models import Sequential, Model
import tensorflow as tf
from keras.applications.efficientnet import (
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB0,
)
from keras.layers import Dense, Lambda, Dot, Activation, Concatenate, LayerNormalization
from keras.layers import Layer
from x3d import X3D
from configs.default import get_default_config


class Attention(Layer):
    def __init__(self, units=128, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def __call__(self, inputs):

        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])

        score_1 = Dense(hidden_size, use_bias=False)(
            hidden_states
        )  # attention_score_vec

        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,))(
            hidden_states
        )  # 'last_hidden_state'
        attn_score = Dot(axes=[1, 2])([h_t, score_1])
        attention_weights = Activation("softmax")(attn_score)

        context_vector = Dot(axes=[1, 1])([hidden_states, attention_weights])

        pre_activation = Concatenate()([context_vector, h_t])
        attention_vector = Dense(self.units, use_bias=False, activation="tanh")(
            pre_activation
        )

        return attention_vector


def cnn(input, shape):
    momentum = 0.9
    NBFRAME = shape[0]
    branch_outputs = []
    for i in range(NBFRAME):
        # Slicing the ith channel:
        out = Lambda(lambda x: x[:, i, :, :], name="Lambda_" + str(i))(input)

        branch_outputs.append(out)

    #     # Concatenating together the per-channel results:
    input_c = Concatenate(axis=-1)(branch_outputs)
    # #input_c = concatenate( [input[i,:,:, :] for i in range(NBFRAME)], axis=0)#tf.keras.layers.Concatenate( tf.unstack(input, axis=1) ) #,

    # x = Conv2D(128, 3, activation="relu")(input_c)
    # x = Conv2D(64, 3, activation="relu")(x)
    # block_1_output = MaxPooling2D(3)(x)
    #
    # x = Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    # x = Conv2D(64, 3, activation="relu", padding="same")(x)
    # block_2_output = add([x, block_1_output])
    #
    # x = Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    # x = Conv2D(64, 3, activation="relu", padding="same")(x)
    # block_3_output = add([x, block_2_output])
    #
    # x = Conv2D(64, 3, activation="relu")(block_3_output)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(256, activation="relu")(x)
    # x = Dropout(0.2)(x)
    # outputs = Dense(2)(x)
    #
    # model = keras.Model(input, outputs, name="toy_resnet")
    # model.summary()

    x = Conv2D(128, (5, 5), input_shape=shape, padding="same", activation="relu")(
        input_c
    )
    x = Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = BatchNormalization(momentum=momentum)(x)

    x = MaxPool2D()(x)

    x = Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = BatchNormalization(momentum=momentum)(x)

    x = MaxPool2D()(x)

    x = Conv2D(256, (5, 5), padding="same", activation="relu")(x)
    x = Conv2D(256, (5, 5), padding="same", activation="relu")(x)
    x = BatchNormalization(momentum=momentum)(x)

    x = MaxPool2D()(x)

    x = Conv2D(512, (5, 5), padding="same", activation="relu")(x)
    x = Conv2D(512, (5, 5), padding="same", activation="relu")(x)
    x = BatchNormalization(momentum=momentum)(x)
    #
    # # base_model = InceptionV3(weights= None, include_top=False).layers.pop(0)
    # # x = Conv2D(64, (3, 3), input_shape=shape,
    # #                   padding='same', activation='relu')(out)
    # # x = base_model(x)
    #
    # # flatten...
    x = GlobalMaxPool2D()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(.2)(x)
    # x = Dense(64, activation='relu')(x)
    output = Dense(2, activation="softmax")(x)

    return output


def cnn_rnn(input, shape):
    # x = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'), input_shape=shape)(input)
    # x = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))(x)
    # x = TimeDistributed(MaxPooling2D(2, 2))(x)
    # x = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))(x)
    # x = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))(x)
    # x = TimeDistributed(MaxPooling2D(2, 2))(x)
    # x = TimeDistributed(BatchNormalization())(x)

    base_model = EfficientNetB0(weights="imagenet", include_top=False)

    # for layer in base_model.layers:
    #     layer.trainable = False

    x = TimeDistributed(base_model)(input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(Flatten())(x)
    x = Dropout(0.2)(x)
    x1 = TimeDistributed(Dense(128, activation="relu"))(x)

    # x1 = LayerNormalization()(x1)
    # x2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.1))(x1)
    # x2 = LSTM(64, return_sequences=True, dropout=0.1)(x1)
    # x = add([x1, x2])
    # x = GRU(32, return_sequences=True, dropout=0.)(x)
    #

    x = Attention(64)(x1)
    # x = GlobalAveragePooling1D()(x1)
    # x= Dense(64, activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dropout(0.1)(x)
    output = Dense(2, activation="softmax")(x)

    return output


def conv_lstm(input, shape):
    x = ConvLSTM2D(
        filters=3,
        kernel_size=(3, 3),
        return_sequences=False,
        data_format="channels_last",
        padding="same",
        input_shape=shape,
    )(input)
    # x = ConvLSTM2D(filters=64, kernel_size=(5, 5), return_sequences=False, data_format="channels_last",

    #              input_shape=shape)(x)
    base_model = EfficientNetB1(weights="imagenet", include_top=False)

    x = base_model(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    output = Dense(2, activation="softmax")(x)

    return output


def CNN3D(input, shape):

    # inputs = Input((width, height, depth, 1))
    # base_model = Inception_Inflated3d(include_top=False, input_shape=shape, weights='rgb_kinetics_only')
    # x = base_model(input)
    # x = GlobalAveragePooling3D()(x)
    # x = Conv3D(filters=64, kernel_size=3, activation="relu")(input)
    # x = MaxPool3D(pool_size=2)(x)
    # x = BatchNormalization()(x)
    #
    # x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    # x = MaxPool3D(pool_size=2)(x)
    # x = BatchNormalization()(x)
    #
    # x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    # x = MaxPool3D(pool_size=2)(x)
    # x = BatchNormalization()(x)
    #
    # x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    # x = MaxPool3D(pool_size=2)(x)
    # x = BatchNormalization()(x)
    cfg = get_default_config()
    cfg.merge_from_file(f"X3D_M.yaml")
    cfg.freeze()

    # input = Reshape((4,16,224,224,3))(input)
    # input1 = tf.keras.layers.Lambda(lambda x: x[0,:,:, :, :])(input)
    base_model = X3D(cfg)  # .call(input1)
    # base_model.load_weights('./checkpoints/model')
    # base_model.pop()
    # base_model.pop()
    # base_model.add(Dense(
    #         units=9,
    #         use_bias=True,
    #         name='fc_2_new',
    #         ))
    # base_model.add(Softmax(dtype='float32'))

    # x = TimeDistributed(base_model)(input1)
    # x = TimeDistributed(GlobalAveragePooling2D())(x)
    # x = TimeDistributed(Flatten())(x)
    # x = Dropout(0.2)(x)
    # x1 = TimeDistributed(Dense(128, activation='relu'))(x)

    # x = GlobalAveragePooling3D()(x)
    # x = Dense(units=512, activation="relu")(x)
    # x = Dropout(0.1)(x)

    # x = Attention(64)(x1)
    # outputs = Dense(2, activation='softmax')(x)#Dense(units=2, activation="softmax")(x)

    # Define the model.
    # model = keras.Model(inputs, outputs, name="3dcnn")
    return base_model
