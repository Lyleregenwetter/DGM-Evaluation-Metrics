import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tqdm.autonotebook import tqdm, trange
import itertools
from matplotlib import pyplot as plt

import pandas as pd
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import evaluation

class discriminator(keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        
        self.Dense1 = keras.layers.Dense(100)
        self.LReLU1 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense2 = keras.layers.Dense(100)
        self.LReLU2 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense3 = keras.layers.Dense(1)
    
    def call(self, inputs):
        
        x = self.Dense1(inputs)
        x = self.LReLU1(x)
        
        x = self.Dense2(x)
        x = self.LReLU2(x)
        
        x = self.Dense3(x)
        
        return x

class generator(keras.Model):
    def __init__(self, data_dim):
        super(generator, self).__init__()
        
        self.Dense1 = keras.layers.Dense(100)
        self.LReLU1 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense2 = keras.layers.Dense(100)
        self.LReLU2 = keras.layers.LeakyReLU(alpha = 0.2)
        
#         self.Dense3 = keras.layers.Dense(128)
#         self.LReLU3 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense4 = keras.layers.Dense(data_dim)
#         self.out_activation = keras.layers.Activation(keras.activations.tanh)
        
    def call(self, inputs):
        
        x = self.Dense1(inputs)
        x = self.LReLU1(x)
        
        x = self.Dense2(x)
        x = self.LReLU2(x)
        
#         x = self.Dense3(x)
#         x = self.LReLU3(x)
        
        x = self.Dense4(x)
#         x = self.LReLU2(x)
        
        return x

class PadGAN(object):
    def __init__(self, conditional, noise_dim = 2, data_dim = 2, lambda0 = 2.0, lambda1 = 0.5, numgen=None, condition=None):

        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.def_numgen = numgen
        self.def_condition=condition
        self.EPSILON = 1e-5
        self.train_conditional = conditional
        self.discriminator = discriminator()

        self.generator = generator(data_dim)
    def compute_diversity_loss(self, x, y):
        
        r = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
        D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5 * tf.math.square(D))
        y = tf.squeeze(y)
        
        if self.lambda0 == 'inf':

            eig_val, _ = tf.linalg.eigh(S)
            loss = -10 * tf.reduce_mean(y)

            Q = None
            L = None

        elif self.lambda0 == 'naive':

            eig_val, _ = tf.linalg.eigh(S)
            loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(eig_val, self.EPSILON))) - 10 * tf.reduce_mean(y)

            Q = None
            L = None

        else:
            Q = tf.tensordot(tf.expand_dims(y, 1), tf.expand_dims(y, 0), 1)
            if self.lambda0 == 0.:
                L = S
            else:
                L = S * tf.math.pow(Q, self.lambda0)
            L = tf.math.minimum(L, self.EPSILON) #adding this for stability
            try:
                eig_val, _ = tf.linalg.eigh(L)
            except: 
                eig_val = tf.ones_like(y)
            loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(eig_val, self.EPSILON)))

        return loss, D, S, Q, L, y
    
    #Simple function to concatenate conditioning vectors to data
    def append_conditioning(self, x, c):
        c = tf.cast(c, "float32")
        c = tf.transpose(tf.expand_dims(c, 0))
        res = tf.concat([x, c], axis=1)
        return res
    
    @tf.function
    def train_step(self, X_batch, c_batch, equation, d_optimizer, g_optimizer):
        binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        batch_size = X_batch.shape[0]
        
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        if self.train_conditional:
            z = self.append_conditioning(z, c_batch)
            
        X_fake = self.generator(z)
        if self.train_conditional:
            X_fake = self.append_conditioning(X_fake, c_batch)
            X_batch = self.append_conditioning(X_batch, c_batch)
            
        with tf.GradientTape() as tape:
            d_real = self.discriminator(X_batch)
            d_loss_real = binary_cross_entropy_loss_fn(tf.ones_like(d_real),d_real)
        variables = self.discriminator.trainable_weights
        gradients = tape.gradient(d_loss_real, variables)
        d_optimizer.apply_gradients(zip(gradients, variables))
            
        with tf.GradientTape() as tape:
            d_fake = self.discriminator(X_fake)
            d_loss_fake = binary_cross_entropy_loss_fn(tf.zeros_like(d_fake),d_fake)
        variables = self.discriminator.trainable_weights
        gradients = tape.gradient(d_loss_fake, variables)
        d_optimizer.apply_gradients(zip(gradients, variables))
        
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        if self.train_conditional:
            z = self.append_conditioning(z, c_batch)
        
        with tf.GradientTape() as tape:
            x_fake_train = self.generator(z)
            if self.train_conditional:
                  x_fake_train = self.append_conditioning(x_fake_train, c_batch)
            d_fake = self.discriminator(x_fake_train)
            g_loss = binary_cross_entropy_loss_fn(tf.ones_like(d_fake),d_fake)

            if not self.lambda1==0:
                y = tf.expand_dims(equation(x_fake_train),-1)
                dpp_loss = self.compute_diversity_loss(x_fake_train, y)[0]
                g_dpp_loss = g_loss + self.lambda1 * dpp_loss
            else:
                y = 0
                dpp_loss = 0
                g_dpp_loss = g_loss
        
        variables = self.generator.trainable_weights
        gradients = tape.gradient(g_dpp_loss, variables)
        g_optimizer.apply_gradients(zip(gradients, variables))
        
        mean_y = tf.reduce_mean(y)
        
        return d_loss_real, d_loss_fake, g_loss, dpp_loss, mean_y
    
    def train(self, X, C, num_anim, equation, steps=10000, batch_size=8, disc_lr=1e-3, gen_lr=1e-3):
        
        if num_anim: #For use when saving generated samples at intermediate steps in training
            results = [] #List of generated arrays at various steps in the training process
            checkpoint_steps=[] #Epoch counts corresponding to the generated arrays
        
        g_optimizer = keras.optimizers.Adam(learning_rate = gen_lr, beta_1=0.5)
        d_optimizer = keras.optimizers.Adam(learning_rate = disc_lr, beta_1=0.5)
        
        steps_range = trange(steps, desc='GAN Training:', leave=True, ascii ="         =")
        
        for step in steps_range:
            ind = np.random.choice(X.shape[0], size=batch_size, replace=False)
            X_batch = tf.cast(X[ind],tf.float32)
            C_batch = tf.cast(C[ind],tf.float32)
            d_loss_real, d_loss_fake, g_loss, dpp_loss, mean_y = self.train_step(X_batch, C_batch, equation, d_optimizer, g_optimizer)
            
            steps_range.set_postfix_str(
                    "[D] R = %+.7f, F = %+.7f [G] F = %+.7f, dpp = %+.7f,  y = %+.7f" % (d_loss_real, d_loss_fake,
                                                                                         g_loss, dpp_loss, mean_y))
            if num_anim and ((step+1)*num_anim)%steps<num_anim:
                result = self.generate(self.def_numgen, self.def_condition)
                results.append(result)
                checkpoint_steps.append(step+1)
        if num_anim:
            results = np.stack(results)
            checkpoint_steps = np.array(checkpoint_steps)
            return results, checkpoint_steps
        else:
            return self
    def generate(self, num, c):
        z = tf.random.normal((num, self.noise_dim), stddev = 0.5, dtype=tf.float32)
        if self.train_conditional:
            z = self.append_conditioning(z, c)
        x_fake = self.generator(z, training=False)
        return x_fake
    
def evaluate_MO(y):
#     ref = np.quantile(np.array(y), 0.5, axis=0)
#     ref = tf.constant(ref, dtype="float32")
#     y=tf.math.maximum(y, 0.0)
#     y=tf.divide(ref,(ref+y))
#     slicelist=[]
    mask=tf.random.uniform(tf.shape(y), minval=0, maxval=1)
    masksums=tf.reduce_sum(mask, axis=1)
    masksums=tf.reshape(masksums, (-1, 1))
    mask=tf.divide(mask, masksums)
    scores=tf.multiply(y, mask)
    scores=tf.reduce_sum(scores, axis=1)
    return scores

def tf_inv_scale(y_scaled, y_scaler):
    scale=tf.cast(tf.convert_to_tensor(y_scaler.scale_), "float32")
    mean=tf.cast(tf.convert_to_tensor(y_scaler.mean_), "float32")
    return tf.cast(y_scaled*scale+mean, "float32")

def pad_aux_loss(x, regressor, classifier, scaler, eval_func):
    y_pred = regressor(x)
    if scaler != None:
        y_pred = tf_inv_scale(y_pred, scaler)
    perfscore = eval_func(y_pred)
    classes = classifier(x)
    return tf.multiply(perfscore, tf.reduce_sum(classes, axis=1))

def clf_trivial(x):
    return tf.ones((len(x), 1))

def create_dnn(ddims, ydims, DO, layers, layersize, batchnorm, activation, r_c): #DNN Builder function
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=ddims))

    for i in range(layers): #Add a dense layer for each layer in range(layers)
        model.add(tf.keras.layers.Dense(layersize)) #Dense layer with size equal to layersize
        if batchnorm=="True": # Add batchnorm if true
            model.add(tf.keras.layers.BatchNormalization())
        if activation=="ReLU": # Add relu or Leaky ReLU
            model.add(tf.keras.layers.ReLU())
        if activation=="Leaky ReLU":
            model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(DO)) #Add dropout with strength equal to DO
    if r_c == "r":
        model.add(tf.keras.layers.Dense(ydims))
    else:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #Sigmoid activation for classification problems
    return model

def fit_classifier(X,N, clfparams):
    X = np.array(X)
    N = np.array(N)
    xdata = np.vstack((X,N))
    ydata = np.hstack((np.ones(len(X)), np.zeros(len(N))))
    x_train, x_val, y_train, y_val = train_test_split(xdata, ydata, test_size=0.2, random_state=1)
    dropout, layers, layersize, batchnorm, activation, patience, lr, batchsize, epochs = clfparams
    model=create_dnn(len(x_train[0,:]), 1, dropout, layers, layersize, batchnorm, activation, "c")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batchsize, validation_data=(x_val, y_val), callbacks=[callback], verbose=1)
    preds = model.predict(x_val)
    f1=sklearn.metrics.f1_score(y_val, np.rint(preds))
    print("Fit classifier with F1: " + str(f1))
    return model
def fit_regressor(X,Y, regparams):
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=1)
    dropout, layers, layersize, batchnorm, activation, patience, lr, batchsize, epochs= regparams
    model=create_dnn(len(x_train[0,:]), len(y_train[0,:]), dropout, layers, layersize, batchnorm, activation, "r")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.MeanSquaredError(), metrics=['MSE'])
    model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batchsize, validation_data=(x_val, y_val), callbacks=[callback], verbose=1)
    preds = model.predict(x_val)
    r2=sklearn.metrics.r2_score(y_val, preds)
    print("Fit regressor with R2: " + str(r2))
    return model

def train_padgan(X, N, Y, C, numgen, numanim, condition, config_params=None, train_params=None, DTAI_params=None, classifier_params=None, regressor_params=None):
    scaling, DTAI, CLF, classifier, regressor, conditional= config_params
    batch_size, disc_lr, gen_lr, lambda0, lambda1, noise_dim, steps= train_params
    ref, p_, a_= DTAI_params
    if scaling:
        y_scaler = StandardScaler()
        y_scaler.fit(Y)
        y_scaled = y_scaler.transform(Y)
    else:
        y_scaler = None
        y_scaled = Y
    if DTAI:
        if ref=="auto":
            ref = np.quantile(Y, 0.75, axis=0)
        if p_ == "auto":
            p_ = np.ones(len(ref))
        if a_ == "auto":
            a_ = 4*np.ones(len(ref))
        def DTAI_wrap(y):
            DTAI_function = evaluation.DTAI_wrapper(ref, p_, a_)
            res, _ = DTAI_function(None, y, None, None, None, False, ref, p_, a_, 1e-7)
            return res
        eval_func = DTAI_wrap
    else:
        eval_func = evaluate_MO
    if lambda1==0:
        print("Lambda1 set to 0, DPP loss disabled; Ignoring CLF and REG...")
        classifier = None
        regressor = None
    else:
        if CLF: 
            if classifier == "auto":

                print("No classifier provided! Fitting DNN Classifier using provided Parameters...")
                classifier = fit_classifier(X,N, classifier_params)
        else: 
            classifier = clf_trivial
        if regressor == "auto":
                print("No regressor provided! Fitting DNN Regressor using provided Parameters...")
                regressor = fit_regressor(X,y_scaled, regressor_params)
    
    #If C is none, set to X to avoid issues. It will be unused
    if C is None:
        if conditional!=0:
            raise Exception("Training in conditional mode but no conditioning data supplied!")
        C=X
        
    model = PadGAN(conditional, lambda0=lambda0, lambda1=lambda1, numgen=numgen, condition=condition, noise_dim = noise_dim, data_dim = len(X[0,:]))
    def pad_aux_loss_wrapper(X):
        return pad_aux_loss(X, regressor, classifier, y_scaler, eval_func)
    return model.train(X, C, numanim, pad_aux_loss_wrapper, steps=steps, batch_size=batch_size, disc_lr=disc_lr, gen_lr=gen_lr)
    
def padgan_wrapper(config_params=None, train_params=None, DTAI_params=None, classifier_params=None, regressor_params=None):
    def model(X, N, Y, C, numgen=None, numanim=None, condition=None):
        return train_padgan(X, N, Y, C, numgen=numgen, numanim=numanim, condition=condition, config_params=config_params, train_params=train_params, DTAI_params=DTAI_params, classifier_params=classifier_params, regressor_params=regressor_params)
    return model
  