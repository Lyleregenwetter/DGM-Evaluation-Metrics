from tensorflow import keras
from tensorflow.keras import layers
from tqdm.autonotebook import tqdm, trange
import numpy as np
import tensorflow as tf

import plotutils

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(object):
    def __init__(self, datadims, conddims, conditional, latent_dim, kl_weight, learning_rate, numgen, condition, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        self.def_numgen = numgen
        self.def_condition=condition
        self.dim_x = datadims
        self.dim_cond = conddims
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.encoder=self.build_encoder()
        self.decoder=self.build_decoder()
        self.train_conditional = conditional
        
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
    
    def train_step(self, data, cond):
        
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
        allweights = self.encoder.trainable_weights + self.decoder.trainable_weights
        grads = tape.gradient(total_loss, allweights)
        self.optimizer.apply_gradients(zip(grads, allweights))
        return total_loss, reconstruction_loss, kl_loss 
        
    def train(self, X, C, num_anim, steps=10000, batch_size=8, disc_lr=1e-3, gen_lr=1e-3, savedir=None):
        
        if num_anim: #For use when saving generated samples at intermediate steps in training
            results = [] #List of generated arrays at various steps in the training process
            checkpoint_steps=[] #Epoch counts corresponding to the generated arrays
        
        
        steps_range = trange(steps, desc='VAE Training:', leave=True, ascii ="         =")
        all_status = []
        for step in steps_range:
            ind = np.random.choice(X.shape[0], size=batch_size, replace=False)
            X_batch = tf.cast(X[ind],tf.float32)
            C_batch = tf.cast(C[ind],tf.float32)
            loss, reconstruction_loss, kl_loss = self.train_step(X_batch, C_batch)
            status = [["L_R", reconstruction_loss], ["L_KL", kl_loss], ["L_tot", loss]]
            all_status.append(status)
            steps_range.set_postfix_str(
                    "L = %+.7f, Reconstruction = %+.7f, KL = %+.7f" % (loss, reconstruction_loss, kl_loss))
            if num_anim and ((step+1)*num_anim)%steps<num_anim:
                result = self.generate(self.def_numgen, self.def_condition)
                results.append(result)
                checkpoint_steps.append(step+1)

        if savedir:
            self.encoder.save(f"{savedir}_encoder")
            self.decoder.save(f"{savedir}_decoder")
            plotutils.trainingplots(all_status, f"{savedir}_training_plot")

        if num_anim:
            results = np.stack(results)
            checkpoint_steps = np.array(checkpoint_steps)
            return results, checkpoint_steps
        else:
            return self
        
    def generate(self, num, c):
        #Sample latent vector
        z = tf.random.normal((num, self.latent_dim), stddev = 0.5, dtype=tf.float32)
        
        #If conditional, append condition to latent vector
        if self.train_conditional:
                z = self.append_conditioning(z, c)
                
        #Call decoder and return constructed sample
        return self.decoder(z)

#Takes 4 inputs: X is real data, N is invalid data (unused), Y is performance values (unused), C is conditioning data
def train_VAE(X, N, Y, C, numgen, numanim, condition, train_params=None, savedir=None):
    #Unpack parameters
    epochs, batch_size, learning_rate, latent_dim, kl_weight, conditional= train_params
    #Initialize and Compile model
    if conditional:
        V = VAE(len(X[0]), 1, conditional, latent_dim, kl_weight, learning_rate, numgen, condition)
    else:
        V = VAE(len(X[0]), 0, conditional, latent_dim, kl_weight, learning_rate, numgen, condition)
    
    #If C is none, set to X to avoid issues. It will be unused
    if C is None:
        if conditional!=0:
            raise Exception("Training in conditional mode but no conditioning data supplied!")
        C=X
    
    #Fit VAE
    return V.train(X, C, numanim, batch_size = batch_size, steps=epochs, savedir=savedir)

#Wrapper function for VAE to be able to call in loop with other models

def VAE_wrapper(train_params=None):
    def model(X, N, Y, C, numgen=None, numanim=None, condition=None, savedir=None):
        return train_VAE(X, N, Y, C, numgen=numgen, numanim=numanim, condition=condition, train_params=train_params, savedir=savedir)
    return model