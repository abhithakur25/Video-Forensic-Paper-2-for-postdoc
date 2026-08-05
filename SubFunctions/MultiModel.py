"""Trainable multi-model comparison on Paper 2 Features.pkl tensors.

All models train/predict on the same (x_train, y_train) / (x_test, y_test)
arrays of shape (N, 15, 32, 32, 10). This is real training — not ResultsP1 CSV
lookup.

Models
------
* DCNN            — lightweight Conv2D stack on time-mean (32,32,10)
* EfficientNetV2B0 — **latest** Keras Applications backbone (newer than B0)
                    used in Paper-1-style baselines; weights=None so the run
                    does not require ImageNet download on restricted networks
* MobileNetV2     — additional modern backbone for coverage
* OM2AHL-BiG      — proposed BiLSTM + GBM (Network.BiLSTMGBM), CoSH optional

Each `train_predict_*` returns integer class predictions of length len(y_test).
"""
from __future__ import annotations

import numpy as np
from keras.layers import (
    Activation, BatchNormalization, Bidirectional, Conv2D, Dense, Dropout,
    Flatten, GlobalAveragePooling2D, Input, LSTM, MaxPooling2D,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2

from SubFunctions.feature_adapters import as_rgb_image, as_spatial


LATEST_BACKBONE = "EfficientNetV2B0"
LATEST_BACKBONE_REASON = (
    "EfficientNetV2B0 is the V2 family successor to EfficientNetB0 used in "
    "historical paper baselines; available in keras.applications under TF 2.10 "
    "without upgrading the project's TensorFlow pin."
)


def _fit_predict_cnn_head(x_tr, y_tr, x_te, epochs, batch_size, lr, name):
    y_cat = to_categorical(y_tr)
    n_class = y_cat.shape[1]
    inp = Input(shape=x_tr.shape[1:])
    x = Conv2D(32, (3, 3), padding="same")(inp)
    x = Activation("relu")(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(n_class, activation="softmax")(x)
    model = Model(inp, out, name=name)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(x_tr, y_cat, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
    pred = np.argmax(model.predict(x_te, verbose=0), axis=1)
    return pred.astype(int)


def train_predict_dcnn(x_train, y_train, x_test, y_test, epochs=3, batch_size=16,
                       learning_rate=0.001):
    """Simple DCNN on time-mean spatial tensors."""
    x_tr = as_spatial(x_train)
    x_te = as_spatial(x_test)
    return _fit_predict_cnn_head(
        x_tr, y_train, x_te, epochs, batch_size, learning_rate, "DCNN_P2")


def train_predict_efficientnetv2(x_train, y_train, x_test, y_test, epochs=3,
                                 batch_size=8, learning_rate=0.001, img_size=64):
    """Latest backbone: EfficientNetV2B0 fine-tuned head on projected RGB maps."""
    x_tr = as_rgb_image(x_train, size=img_size)
    x_te = as_rgb_image(x_test, size=img_size)
    # scale to [0,1] for backbone stability without imagenet preprocess
    x_tr = x_tr / 255.0
    x_te = x_te / 255.0
    y_cat = to_categorical(y_train)
    n_class = y_cat.shape[1]

    base = EfficientNetV2B0(
        include_top=False,
        weights=None,  # no ImageNet download required
        input_shape=(img_size, img_size, 3),
        pooling="avg",
    )
    base.trainable = True
    inp = Input(shape=(img_size, img_size, 3))
    x = base(inp, training=True)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_class, activation="softmax")(x)
    model = Model(inp, out, name="EfficientNetV2B0_P2")
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    model.fit(x_tr, y_cat, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
    pred = np.argmax(model.predict(x_te, verbose=0), axis=1)
    return pred.astype(int)


def train_predict_mobilenetv2(x_train, y_train, x_test, y_test, epochs=3,
                              batch_size=8, learning_rate=0.001, img_size=64):
    """MobileNetV2 modern lightweight backbone on projected RGB maps."""
    x_tr = as_rgb_image(x_train, size=img_size) / 255.0
    x_te = as_rgb_image(x_test, size=img_size) / 255.0
    y_cat = to_categorical(y_train)
    n_class = y_cat.shape[1]

    base = MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(img_size, img_size, 3),
        pooling="avg",
    )
    base.trainable = True
    inp = Input(shape=(img_size, img_size, 3))
    x = base(inp, training=True)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_class, activation="softmax")(x)
    model = Model(inp, out, name="MobileNetV2_P2")
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    model.fit(x_tr, y_cat, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
    pred = np.argmax(model.predict(x_te, verbose=0), axis=1)
    return pred.astype(int)


def train_predict_om2ahl_big(x_train, y_train, x_test, y_test, epochs=3,
                             batch_size=16, learning_rate=0.001, skip_opt=True):
    """Proposed OM2AHL-BiG (BiLSTMGBM). Optionally skip CoSH optimization."""
    import SubFunctions.Model as ModelMod

    if skip_opt:
        class _NoOpt:
            def __init__(self, model, x_test, y_test):
                self.model = model

            def main_update_hyperparameters(self):
                return self.model

        ModelMod.Optimization = _NoOpt

    try:
        import keras.utils as ku
        ku.plot_model = lambda *a, **k: None
    except Exception:
        pass

    net = ModelMod.Network(
        x_train=x_train, x_test=x_test,
        y_train=y_train, y_test=y_test, epochs=epochs,
    )
    net.batch_size = batch_size
    net.learning_rate = learning_rate
    pred = net.BiLSTMGBM(epochs=epochs)
    return np.asarray(pred).astype(int)


# Registry used by the driver / tests
MODEL_REGISTRY = {
    "DCNN": train_predict_dcnn,
    "EfficientNetV2B0": train_predict_efficientnetv2,  # latest backbone
    "MobileNetV2": train_predict_mobilenetv2,
    "OM2AHL-BiG": train_predict_om2ahl_big,
}


def run_model(name, x_train, y_train, x_test, y_test, epochs=3, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; choose from {list(MODEL_REGISTRY)}")
    fn = MODEL_REGISTRY[name]
    # filter kwargs per signature lightly
    if name != "OM2AHL-BiG":
        kwargs.pop("skip_opt", None)
    return fn(x_train, y_train, x_test, y_test, epochs=epochs, **kwargs)
