# This is the old version of the evals file, which was used in the paper
# The TensorFlow dependency has been removed for the new version, but the models still use TF. 

from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import sklearn
import random
from tqdm import trange
import tensorflow as tf
import eval_prd
import importlib
importlib.reload(eval_prd)

#Each of these evlauation metrics are contained in a wrapper function and structured to be called using the 
#same variables so that they can be called in loop. The wrapper function returns the callable action. 
#Inputs are: 
#x_eval = design space values of generated designs, 
#y_eval = performance space values of generate designs, 
#x_data = design space values of original data
#y_data = performance space values of original data
#n_data = design space values of invalid samples
#scorebars = whether to show progress bars and staus updates

#Any constants are passed to the function in optional arguments when the wrapper function is called


def Hypervolume_wrapper(hv_ref="auto"):
    def Hypervolume(x_eval, y_eval, x_data, y_data, n_data, scorebars, hv_ref=hv_ref):
        y_eval = np.array(y_eval)
        if scorebars:
            print("Calculating Hypervolume")
        if hv_ref=="auto":
            hv_ref = np.quantile(y_eval, 0.99, axis=0)
            print("Warning: no reference point provided!")
        hv = HV(ref_point=hv_ref)
        hvol = hv(y_eval)
        return None, hvol
    return Hypervolume

def Generational_distance_wrapper(pf):
    def Generational_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, pf=pf):
        y_eval = np.array(y_eval)
        if scorebars:
            print("Calculating Generational Distance")
        gd = GD(pf)
        hvol = gd(y_eval)
        return None, hvol
    return Generational_distance

def get_perc_band(value, data, band):
    perc = sum(data < value) / len(data)
    if perc < band/2:
        lower = 0
        upper = band
    elif perc > 1-band/2:
        lower = 1-band
        upper = 1
    else:
        lower = perc-band/2
        upper = perc+band/2
    lb = np.quantile(data, lower)
    ub = np.quantile(data, upper)
    mask = np.logical_and(data>=lb, data<=ub)
    return mask
       

def calc_distance(X, Y, distance="Euclidean"):
    if distance=="Euclidean":
        return L2_vectorized(X,Y)
    else:
        raise Exception("Unknown distance metric specified")
        
        
def L2_vectorized(X, Y):
    #Vectorize L2 calculation using x^2+y^2-2xy
    X_sq = np.sum(np.square(X), axis=1)
    Y_sq = np.sum(np.square(Y), axis=1)
    sq = np.add(np.expand_dims(X_sq, axis=-1), np.transpose(Y_sq)) - 2*np.matmul(X,np.transpose(Y))
    sq = np.clip(sq, 0.0, 1e12)
    return np.sqrt(sq)



def signed_distance_to_boundary_wrapper(direction, ref, p_, method="linear"):
    def signed_distance_to_boundary(x_eval, y_eval, x_data, y_data, n_data, scorebars, direction=direction, ref=ref, p_=p_, method=method):
        if method=="linear":
            diff = np.subtract(y_eval,ref)
        elif method=="log":
            diff = np.log(np.divide(y_eval, ref))
        else:
            raise Exception("Unknown method, expected linear or log")
            
        if direction=="maximize":
            pass
        elif direction=="minimize":
            diff=-diff
        else:
            raise Exception("Unknown optimization direction, expected maximize or minimize")
        diff_sc = np.multiply(diff, p_)
        diff_clip = np.minimum(diff_sc, np.zeros_like(diff_sc))
        zeros = np.expand_dims(np.zeros(np.shape(diff)[1]), axis=0)
        dists_clip = L2_vectorized(diff_clip, zeros)
        dists = diff_sc.min(axis=1, keepdims=True)
        dists_mask = tf.reduce_all(tf.math.greater(diff, 0), axis=1)
        dists_mask = tf.expand_dims(tf.cast(dists_mask, "float32"), axis=1)
        final_scores = tf.multiply(dists_mask, dists)-dists_clip
        return final_scores, tf.reduce_mean(final_scores)
    return signed_distance_to_boundary
        
def gen_gen_distance_wrapper(flag, reduction="min", distance="Euclidean"):
    def gen_gen_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag):
        if scorebars:
            print("Calculating Gen-Gen Distance")
        scores = []
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        res = calc_distance(x, x, distance)
        res = tf.linalg.set_diag(res, tf.reduce_max(res, axis=1))
        if reduction == "min":
            scores = tf.reduce_min(res, axis=1)
        elif reduction == "ave":
            scores = tf.reduce_mean(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, tf.reduce_mean(scores)
    return gen_gen_distance

def distance_to_centroid_wrapper(flag, distance="Euclidean"):
    def distance_to_centroid(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag):
        if scorebars:
            print("Calculating Distance to Centroid")
        scores = []
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        centroid = np.mean(x, axis=0)
        vec = np.subtract(x, centroid)
        distance = np.linalg.norm(vec, axis=1)
        return scores, tf.reduce_mean(distance)
    return distance_to_centroid


def DPP_diversity_wrapper(flag, subset_size=10):
    def DPP_diversity(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag):
        # Average log determinant
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        x = tf.convert_to_tensor(x, dtype="float32")
        r = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
        D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5 * tf.math.square(D))
        y = tf.ones(np.shape(x)[0])
        try:
            eig_val, _ = tf.linalg.eigh(S)
        except: 
            eig_val = tf.ones_like(y)
        loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(eig_val, 1e-7)))
        return None, loss
    return DPP_diversity



def gen_data_distance_wrapper(flag, reduction="min", distance="Euclidean"):
    def gen_data_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag = flag):
        if scorebars:
            print("Calculating Gen-Data Distance")
        scores = []
        
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        num_eval = np.shape(x)[0]
        res = calc_distance(x, data, distance)

        if reduction == "min":
            scores = tf.reduce_min(res, axis=1)
        elif reduction == "ave":
            scores = tf.reduce_mean(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, tf.reduce_mean(scores)
    return gen_data_distance

def data_gen_distance_wrapper(flag, reduction="min", distance="Euclidean"):
    def data_gen_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag = flag):
        if scorebars:
            print("Calculating Data-Gen Distance")
        scores = []
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            y_eval = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
            
        num_eval = np.shape(x)[0]
        res = calc_distance(data, x, distance)

        if reduction == "min":
            scores = tf.reduce_min(res, axis=1)
        elif reduction == "ave":
            scores = tf.reduce_mean(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return None, tf.reduce_mean(scores)
    return data_gen_distance

def DTAI_wrapper(direction, ref, p_, a_):
    def DTAI(x_eval, y_eval, x_data, y_data, n_data, scorebars, direction=direction, ref=ref, p_=p_, a_=a_, DTAI_EPS=1e-7):
        p_ = tf.cast(p_, "float32")
        a_ = tf.cast(a_, "float32")
        y_eval = tf.cast(y_eval, "float32")
        if scorebars:
            print("Calculating DTAI")

        #y values must be greater than 0
        y=tf.math.maximum(y_eval, DTAI_EPS)
        if direction=="maximize":
            x=tf.divide(y, ref)
        elif direction=="minimize":
            x = tf.divide(ref, y)
        else:
            raise Exception("Unknown optimization direction, expected maximize or minimize")
        case1 = tf.multiply(p_,x)-p_
        p_over_a=tf.divide(p_,a_)
        exponential=tf.exp(tf.multiply(a_, (1-x)))
        case2=tf.multiply(p_over_a, (1-exponential))
        casemask = tf.greater(x, 1)
        casemask = tf.cast(casemask, "float32")
        scores=tf.multiply(case2, casemask) + tf.multiply( case1, (1 - casemask))
        scores=tf.math.reduce_sum(scores, axis=1)         
        smax=tf.math.reduce_sum(p_/a_)
        smin=-tf.math.reduce_sum(p_)

        scores=(scores-smin)/(smax-smin)
        return scores, tf.reduce_mean(scores)
    return DTAI
# def minimum_target_ratio_wrapper(direction, ref):
#     def minimum_target_ratio(x_eval, y_eval, x_data, y_data, n_data, scorebars, direction=direction, ref=ref):
#         if scorebars:
#             print("Calculating Minimum Target Ratio")
#         y_eval = tf.cast(y_eval, "float32")
#         ref = tf.cast(ref, "float32")
#         if direction=="maximize":
#             res = tf.divide(y_eval, ref)
#         elif direction=="minimize":
#             res = tf.divide(ref, y_eval)
#         else:
#             raise Exception("Unknown optimization direction, expected maximize or minimize")
#         scores = tf.reduce_min(res, axis=1)
#         return scores, tf.reduce_mean(scores)
#     return minimum_target_ratio

def weighted_target_success_rate_wrapper(direction, ref, p_):
    def weighted_target_success_rate(x_eval, y_eval, x_data, y_data, n_data, scorebars, direction=direction, ref=ref, p_=p_):
        y_eval = tf.cast(y_eval, "float32")
        if scorebars:
            print("Calculating Weighted Target Success Rate")
        num_eval = y_eval[:,0]
        y_eval = tf.cast(y_eval, "float32")
        ref = tf.cast(ref, "float32")
        p_ = tf.cast(p_, "float32")
        if direction=="maximize":
            res = tf.cast(y_eval>ref, "float32")
        elif direction=="minimize":
            res = tf.cast(y_eval<ref, "float32")
        else:
            raise Exception("Unknown optimization direction, expected maximize or minimize")
        scores = res
        scaled_scores = tf.matmul(scores, tf.expand_dims(p_, -1))/sum(p_)
        return scaled_scores, tf.reduce_mean(scaled_scores)
    return weighted_target_success_rate

def gen_neg_distance_wrapper(reduction = "min", distance="Euclidean"):
    def gen_neg_distance(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating Gen-Neg Distance")
        res = calc_distance(x_eval, n_data, distance)
        if reduction == "min":
            scores = tf.reduce_min(res, axis=1)
        elif reduction == "ave":
            scores = tf.reduce_mean(res, axis=1)
        else:
            raise Exception("Unknown reduction method")
        return scores, tf.reduce_mean(scores)
    return gen_neg_distance

def MMD_wrapper(flag, sigma=1, batch_size=1000, num_iter=100, biased=True):
    def MMD(x_eval, y_eval, x_data, y_data, n_data, scorebars, flag=flag, sigma=sigma, biased=biased):
        if scorebars:
                print("Calculating Maximum Mean Discrepancy")
        if flag == "x":
            x = x_eval
            data = x_data
        elif flag == "y":
            x = y_eval
            data = y_data
        elif flag == "all":
            y_eval = pd.concat([x_eval, y_eval], axis=0)
            data = pd.concat([x_data, y_data], axis=0)
        else:
            raise Exception("Unknown flag passed")
        total=0
        for i in range(num_iter):
            if len(x) > batch_size:
                X = x[np.random.randint(x.shape[0], size=batch_size), :]    
            else:
                X = x
            if len(data) > batch_size:
                Y = data[np.random.randint(data.shape[0], size=batch_size), :]
            else:
                Y = data
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            Y = tf.convert_to_tensor(Y, dtype=tf.float32)
            gamma = 1 / (2 * sigma**2)
        
            XX = tf.matmul(X, tf.transpose(X))
            XY = tf.matmul(X, tf.transpose(Y))
            YY = tf.matmul(Y, tf.transpose(Y))
        
            X_sqnorms = tf.linalg.diag_part(XX)
            Y_sqnorms = tf.linalg.diag_part(YY)
        
            K_XY = tf.math.exp(-gamma * (
                    -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        #             -2 * XY + tf.expand_dims(X_sqnorms, 1) + tf.expand_dims(Y_sqnorms, 0)))
        
            K_XX = tf.math.exp(-gamma * (
                    -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
            K_YY = tf.math.exp(-gamma * (
                    -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
            
            if biased:
                mmd2 = tf.math.reduce_mean(K_XX) + tf.math.reduce_mean(K_YY) - 2 * tf.math.reduce_mean(K_XY)
            else:
                m = K_XX.shape[0]
                n = K_YY.shape[0]
        
                mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
                    + (K_YY.sum() - n) / (n * (n - 1))
                    - 2 * K_XY.mean())
            total+=mmd2
        return None, total.numpy()/num_iter
    return MMD

def F_wrapper(flag, beta=1, num_clusters=20, num_angles=1001, num_runs=5, enforce_balance=False):
    def calc_prd(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating F" + str(beta))
        if os.path.isfile(f"temp_eval_recall_{flag}.npy") and os.path.isfile(f"temp_eval_precision_{flag}.npy"):
            recall = np.load(f"temp_eval_recall_{flag}.npy")
            precision = np.load(f"temp_eval_precision_{flag}.npy")
        else:
            if flag == "x":
                x = x_eval
                data = x_data
            elif flag == "y":
                x = y_eval
                data = y_data
            elif flag == "all":
                x = pd.concat([x_eval, y_eval], axis=0)
                data = pd.concat([x_data, y_data], axis=0)
            else:
                raise Exception("Unknown flag passed")
            recall, precision = eval_prd.compute_prd_from_embedding(x, data, num_clusters=num_clusters, num_angles=num_angles, num_runs=num_runs, enforce_balance=enforce_balance)
            np.save(f"temp_eval_recall_{flag}.npy", recall)
            np.save(f"temp_eval_precision_{flag}.npy", precision)
        F = eval_prd._prd_to_f_beta(precision, recall, beta=beta, epsilon=1e-10)
        return None, max(F)
    return calc_prd

def AUC_wrapper(flag, num_clusters=20, num_angles=1001, num_runs=5, enforce_balance=False, plot=False):
    def calc_prd(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating AUC")
        if os.path.isfile(f"temp_eval_recall_{flag}.npy") and os.path.isfile(f"temp_eval_precision_{flag}.npy"):
            recall = np.load(f"temp_eval_recall_{flag}.npy")
            precision = np.load(f"temp_eval_precision_{flag}.npy")
        else:
            if flag == "x":
                x = x_eval
                data = x_data
            elif flag == "y":
                x = y_eval
                data = y_data
            elif flag == "all":
                x = pd.concat([x_eval, y_eval], axis=0)
                data = pd.concat([x_data, y_data], axis=0)
            else:
                raise Exception("Unknown flag passed")
            recall, precision = eval_prd.compute_prd_from_embedding(x, data, num_clusters=num_clusters, num_angles=num_angles, num_runs=num_runs, enforce_balance=enforce_balance)
            np.save(f"temp_eval_recall_{flag}.npy", recall)
            np.save(f"temp_eval_precision_{flag}.npy", precision)
        F1 = eval_prd._prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10)
        prd_data = [np.array([precision,recall])]
        if plot:
            eval_prd.plot(prd_data, labels=None, out_path=None,legend_loc='lower left', dpi=300)
        tot = 0
        for i in range(len(precision)-1):
            tot+=(precision[i]+precision[i+1])/2*(recall[i+1]-recall[i])
        
        return None, tot
    return calc_prd

def evaluate_validity(x_fake, validityfunction):
    scores = validityfunction(x_fake)
    return scores, np.mean(scores)

def convex_hull_wrapper(flag):
    def convex_hull(x_eval, y_eval, x_data, y_data, n_data, scorebars):
        if scorebars:
            print("Calculating Convex Hull")
        if flag == "x":
            x = x_eval
        elif flag == "y":
            x = y_eval
        elif flag == "all":
            x = pd.concat([x_eval, y_eval], axis=0)
        else:
            raise Exception("Unknown flag passed")
        hull = ConvexHull(x)
        return None, hull.volume
    return convex_hull


#In this metric, we assume the conditioning information is instead passed in as the performance space values (y)
def predicted_conditioning_wrapper(reg, cond):
    def predicted_conditioning(x_eval, y_eval, x_data, y_data, n_data, scorebars, reg=reg, cond=cond):
        if scorebars:
            print("Calculating predicted_constraint_satisfaction")
        c_data = y_data
        reg.fit(x_data, c_data)
        res = reg.predict(x_eval)
        cond = np.ones_like(res)*cond
        score = sklearn.metrics.mean_squared_error(res, cond)
        return None, score
    return predicted_conditioning



def predicted_constraint_satisfaction_wrapper(clf):
    def predicted_constraint_satisfaction(x_eval, y_eval, x_data, y_data, n_data, scorebars, clf=clf):
        if scorebars:
            print("Calculating predicted_constraint_satisfaction")
        x_all = np.concatenate([x_data, n_data], axis=0)
        y_all = np.concatenate([np.ones(len(x_data)), np.zeros(len(n_data))], axis=0)
        clf.fit(x_all, y_all)
        res= clf.predict_proba(x_eval)[:,1]
        return res, tf.reduce_mean(res)
    return predicted_constraint_satisfaction

#"CLF" may be a classifier or regressor
def ML_efficacy_wrapper(clf, score):
    def ML_efficacy(x_eval, y_eval, x_data, y_data, n_data, scorebars, clf=clf, score=score):
        if scorebars:
            print("Calculating ML Efficacy")
        clf.fit(x_eval, y_eval)
        preds = clf.predict(x_data)
        res = score(y_data, preds)
        
        return res, tf.reduce_mean(res)
    return ML_efficacy


  