from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, datadims, conddims, conditional, latent_dim, kl_weight, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.dim_x = datadims
        self.dim_cond = conddims
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.encoder=self.build_encoder()
        self.decoder=self.build_decoder()
        self.train_conditional = conditional
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    def build_encoder(self): 
        inp = layers.Input(shape=self.dim_x+ self.dim_cond)
        
        x = layers.Dense(100, activation="relu")(inp)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(100, activation="relu")(x)
        
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inp, [z_mean, z_log_var, z], name="encoder")
        return encoder
    
    
    def build_decoder(self):
        inp = layers.Input(shape=self.latent_dim + self.dim_cond)
        x = layers.Dense(100, activation="relu")(inp)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(100, activation="relu")(x)
        out = layers.Dense(self.dim_x)(x)
        decoder = keras.Model(inp, out, name="encoder")
        return decoder
    
    
    #Simple function to concatenate conditioning vectors to data
    def append_conditioning(self, x, c):
        c = tf.cast(c, "float32")
        c = tf.transpose(tf.expand_dims(c, 0))
        res = tf.concat([x, c], axis=1)
        return res
    
    def train_step(self, inp):
        data, cond = inp[0]
        
        #Start gradient tape in which we calculate all losses
        with tf.GradientTape() as tape:
            
            #If training conditional, build concatenated vector of input and conditioning vector
            if self.train_conditional:
                data_cond = self.append_conditioning(data, cond)
            else:
                data_cond = data
                
            #Call encoder 
            z_mean, z_log_var, z = self.encoder(data_cond)
            
            #If training conditional, append condition to latent vector
            if self.train_conditional:
                z = self.append_conditioning(z, cond)
            
            #Call decoder
            reconstruction = self.decoder(z)
            
            #Calculate reconstruction loss
            reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(data, reconstruction))
            
            #Calcualte KL Divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            #Total loss
            total_loss = reconstruction_loss + self.kl_weight*kl_loss
            
        #Apply gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        } 
        
    def generate(self, num, c):
        #Sample latent vector
        z = tf.random.normal((num, self.latent_dim), stddev = 0.5, dtype=tf.float32)
        
        #If conditional, append condition to latent vector
        if self.train_conditional:
                z = self.append_conditioning(z, c)
                
        #Call decoder and return constructed sample
        return self.decoder(z)

#Takes 4 inputs: X is real data, N is invalid data (unused), Y is performance values (unused), C is conditioning data
def train_VAE(X, N, Y, C, train_params=None):
    #Unpack parameters
    epochs, batch_size, learning_rate, latent_dim, kl_weight, conditional= train_params
    #Initialize and Compile model
    if conditional:
        V = VAE(len(X[0]), 1, conditional, latent_dim, kl_weight)
    else:
        V = VAE(len(X[0]), 0, conditional, latent_dim, kl_weight)
    V.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    #If C is none, set to X to avoid issues. It will be unused
    if C is None:
        if conditional!=0:
            raise Exception("Training in conditional mode but no conditioning data supplied!")
        C=X
    
    #Convert things to float32 tensors
    X=tf.convert_to_tensor(X)
    X = tf.cast(X,tf.float32)
    C=tf.convert_to_tensor(C)
    C = tf.cast(C,tf.float32)
    
    #Fit VAE
    V.fit([X,C], epochs=epochs, batch_size=batch_size)
    
    #Return fitted VAE object
    return V

#Wrapper function for VAE to be able to call in loop with other models

def VAE_wrapper(train_params=None):
    def model(X, N, Y, C):
        return train_VAE(X, N, Y, C, train_params=train_params)
    return model