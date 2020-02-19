
from keras.models import Model
from keras.layers import BatchNormalization, Conv1D, Dense, Input,  TimeDistributed, Activation, Bidirectional, GRU


def rnn_automatic_speech_recognition(
        input_dim=13,
        filters=200,
        kernel_size=11,
        conv_stride=2,
        dilation=1,
        units=200,
        activation="relu",
        dropout=.1,
        recur_layers=3,
        output_dim=29):

    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(
        filters,
        kernel_size,
        strides=conv_stride,
        padding="valid",
        activation=activation)(input_data)

    layer = BatchNormalization()(conv_1d)

    for _ in range(recur_layers):
        layer = Bidirectional(GRU(
            units,
            dropout=dropout,
            activation=activation,
            return_sequences=True,
            implementation=2, ))(layer)

        layer = BatchNormalization()(layer)

    time_dense = TimeDistributed(Dense(output_dim))(layer)

    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)

    def cnn_output_length(input_length, ):
        if input_length is None:
            return None
        dilated_filter_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        output_length = input_length - dilated_filter_size + 1
        return (output_length + conv_stride - 1) // conv_stride

    model.output_length = cnn_output_length

    return model