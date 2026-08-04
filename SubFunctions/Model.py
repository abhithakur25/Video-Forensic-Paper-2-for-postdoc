import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow import keras
from termcolor import cprint
import tensorflow as tf
from keras.losses import categorical_crossentropy, CategoricalCrossentropy
from SubFunctions.IncrementalLearning import Incremental_Learning
from SubFunctions.Attention import mutual_attention, SpaSelfAttention, channel_attention_module, ZeroAttention
from SubFunctions.Optimization import Optimization

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=0.05, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return CategoricalCrossentropy()(tf.squeeze(labels), feature_vectors)


class Network:

    def __init__(self, x_train, x_test, y_train, y_test, epochs):
        # Constructor to initialize class attributes.
        self.x_train = x_train  # Training data
        self.x_test = x_test
        self.y_train = y_train  # Training labels
        self.y_test = y_test  # Testing labels
        self.epochs = epochs  # Number of training epochs
        self.batch_size = 32  # Number of training
        self.learning_rate = 0.001  # Number of training

    @staticmethod
    def MultiLevelAttention(x):
        # Multi-level attention module
        mut_cross_att = mutual_attention(x)
        SA = SpaSelfAttention()
        self_att = SA(mut_cross_att)
        return self_att


    @staticmethod
    def MixedAttention(x):
        # Mixed attention module
        x = Reshape(target_shape=(x.shape[1], x.shape[2],  1))(x)
        channe_att = channel_attention_module(x)
        zero_att = ZeroAttention()(x)
        mixed_att = keras.layers.Add()([channe_att, zero_att])
        mixed_att = Reshape(target_shape=(x.shape[1], x.shape[2]))(mixed_att)
        return mixed_att


    def BiLSTMGBM(self, epochs=None):

        cprint("================================", color='magenta')
        cprint("[⚠️] Multilevel mixed Attentional Hybrid learning enabled Bi-LSTM GBM ", 'magenta', on_color='on_grey')
        cprint("================================", color='magenta')

        if epochs is None:
            epochs = self.epochs

        y_train = to_categorical(self.y_train)

        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1] * self.x_train.shape[2], self.x_train.shape[3] * self.x_train.shape[4])
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1] * self.x_test.shape[2], self.x_test.shape[3] * self.x_test.shape[4])


        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))
        x = Bidirectional(LSTM(units=100, return_sequences=True))(input_layer)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Bidirectional(LSTM(units=128, return_sequences=True))(x)
        x = self.MultiLevelAttention(x)
        x = self.MixedAttention(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Bidirectional(LSTM(units=128, return_sequences=False))(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(units=64)(x)
        x = Activation('relu')(x)
        x = Dense(units=32)(x)
        x = Activation('relu')(x)
        output_layer = Dense(y_train.shape[1], activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()


        train_sets = Incremental_Learning(train_data=x_train, train_labels=self.y_train).increment_data()

        for train_set in range(len(train_sets)):
            cprint(f"[⚠️] Increment is {train_set} ", color='grey', on_color='on_red')
            train_data = train_sets[train_set][0]
            train_labels = train_sets[train_set][1]
            train_labels = to_categorical(train_labels)
            model.fit(train_data, train_labels, epochs=epochs, batch_size=self.batch_size, verbose=1, shuffle=True)


        model = Optimization(model, x_test, self.y_test).main_update_hyperparameters()
        train_model = Model(inputs=input_layer, outputs=model.layers[-8].output)

        train_feature = train_model.predict(x_train)
        test_feature = train_model.predict(x_test)

        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                         max_depth=1, random_state=0, verbose=1)


        from keras.utils import plot_model
        plot_model(model, to_file='Results\\Arc.png', show_shapes=True, show_layer_names=True, show_layer_activations=True,
                   show_dtype=True, dpi=1200)

        gb.fit(train_feature, self.y_train)


        # Predict the response for test dataset
        predict = gb.predict(test_feature)

        return predict



