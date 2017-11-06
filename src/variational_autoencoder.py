'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler

def build_model(latent_dim, x_train, x_valid=None, hidden_dims=None, 
                return_dims=None, activation='relu', lr=0.008, epsilon_std=1.0,
                batch_size=20, epochs=1000):
    
    optimizer = optimizers.Adam(lr=lr)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                 patience=40, min_lr=1e-8, 
                                 verbose=1)
    callbacks = [reduce_lr]
    
    original_dim = x_train.shape[-1]
    if hidden_dims is None:
        hidden_dims = int((original_dim-latent_dim)/2)
    
    if not return_dims:
        return_dims = hidden_dims

    if x_valid is not None:
        x_train = x_valid
    
    x = Input(shape=(original_dim,))
    h = Dense(hidden_dims, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
        
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(hidden_dims, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
    
        def vae_loss(self, x, x_decoded_mean):
            xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)
    
        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x
    
    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)
    vae.compile(optimizer=optimizer, loss=None)
    
    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, None),
            callbacks=callbacks)
    
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    
    # build a generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    return encoder, generator
    

def sample_from_generator(history, nb_samples, latent_dim=12, 
                          valid_split=0.3, random_split=True,
                          hidden_dims=None, **kwargs):
    scaler = MinMaxScaler()
    scaler.fit(history)
    scaled = scaler.transform(history)
    
    nb_train = history.shape[0]    
    if not valid_split:
        nb_valid = 0
    elif isinstance(valid_split, float):
        nb_valid = nb_train - int(np.floor(nb_train*valid_split))
    else:
        nb_valid = valid_split
        
    if nb_valid > 0:
        if random_split:
            ind = np.arange(nb_train)
            np.random.shuffle(ind)
            x_valid = scaled[ind[-nb_valid:], :]
            x_train = scaled[ind[:-nb_valid], :]
        else:
            x_valid = scaled[-nb_valid:, :]
            x_train = scaled[:-nb_valid, :]
    else:
        x_valid = None
        x_train = scaled
    
    _, generator = build_model(latent_dim, x_train, x_valid=x_valid, 
                               hidden_dims=hidden_dims, **kwargs)
    
    normal_sample = np.random.standard_normal((nb_samples, latent_dim))
    draws = generator.predict(normal_sample)
    return scaler.inverse_transform(draws)

if __name__ == "__main__":
    #file_name = 'history_205.npy'
    file_name = 'history.npy'
    nb_samples = 10000
    valid_split=0.3
    batch_size = 20
    latent_dim = 16
    epochs = 1000
    epsilon_std = 1.0
    lr = 0.002
    
    history = np.load(file_name)
    draws = sample_from_generator(history, nb_samples, latent_dim=latent_dim, 
                                  valid_split=valid_split, random_split=True,
                                  lr=lr, epsilon_std=epsilon_std, 
                                  batch_size=batch_size, epochs=epochs)