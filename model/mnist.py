from . import abc_model
from . import config

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model


class Mnist(abc_model.ABCModel):
    @classmethod
    def set_callbacks(cls, fname):
        # fname = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
        fpath = config.Config.run_dir_path + "/weight/" + fname
        callbacks = []
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'))

        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, mode='auto'))

        return callbacks

    @classmethod
    def make_model(cls):
        input_layer = Input(shape=(784,))
        layer2 = Dense(512, activation='relu')(input_layer)
        layer2 = Dropout(0.2)(layer2)
        layer3 = Dense(512, activation='relu')(layer2)
        layer3 = Dropout(0.2)(layer3)
        output = Dense(config.Config.num_classes, activation='softmax')(layer3)
        model = Model(input_layer, output)
        model.summary()

        model.compile(loss=config.Config.loss,
                      optimizer=config.Config.optimizer,
                      metrics=[config.Config.metrics])
        return model

    @classmethod
    def save_model(cls, model):
        print("save" + config.Config.save_model)
        model.save(config.Config.save_model)

    @classmethod
    def load_model(cls):
        print("load" + config.Config.save_model)
        from keras.models import load_model
        return load_model(config.Config.save_model)
