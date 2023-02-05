# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:05:43 2022

@author: Lyle
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import tensorflow as tf

def OH_encode(data): 
    data=data.copy()
    #One-hot encode the materials 
    data.loc[:, "Material"]=pd.Categorical(data["Material"], categories=["Steel", "Aluminum", "Titanium"]) 
    mats_oh=pd.get_dummies(data["Material"], prefix="Material=", prefix_sep="") 
    data.drop(["Material"], axis=1, inplace=True) 
    data=pd.concat([mats_oh, data], axis=1) 
    return data 

def load_framed_dataset(c_r="c", onehot = True, scaled = True, augmented = False):
    #key: c=classification, r=regression
    if augmented:
        reg_data = pd.read_csv("all_structural_data_aug.csv", index_col=0)
        clf_data = pd.read_csv("validity_aug.csv", index_col=0)
    else:
        reg_data = pd.read_csv("all_structural_data.csv", index_col=0)
        clf_data = pd.read_csv("validity.csv", index_col=0)
        
    batch = reg_data.iloc[:,-1]
    x_reg = reg_data.iloc[:,:-11]
    x_clf = clf_data.drop(["valid"], axis=1)
    
    if onehot:
        x_reg = OH_encode(x_reg)
        x_clf = OH_encode(x_clf)
    
    if scaled:
        if not onehot:
            x_reg.drop(["Material"])
            x_clf.drop(["Material"])
        scaler = StandardScaler()
        scaler.fit(x_reg)
        x_reg_sc = scaler.transform(x_reg)
        x_clf_sc = scaler.transform(x_clf)
        x_reg = pd.DataFrame(x_reg_sc, columns=x_reg.columns, index=x_reg.index)
        x_clf = pd.DataFrame(x_clf_sc, columns=x_clf.columns, index=x_clf.index)
        if not onehot:
            x_reg["Material"] = reg_data["Material"]
            x_clf["Material"] = clf_data["Material"]
    else:
        scaler = None
    
    if c_r=="c":
        return x_clf, clf_data["valid"], batch, scaler
    else:
        y = reg_data.iloc[:,-11:-1]
        y = modify_framed(y)
        return x_reg, y, batch, scaler
   
def modify_framed(y):
    for col in ['Sim 1 Safety Factor', 'Sim 3 Safety Factor']:
        y[col] = 1/y[col]
        y.rename(columns={col:col+" (Inverted)"}, inplace=True)
    for col in ['Sim 1 Dropout X Disp.', 'Sim 1 Dropout Y Disp.', 'Sim 1 Bottom Bracket X Disp.', 'Sim 1 Bottom Bracket Y Disp.', 'Sim 2 Bottom Bracket Z Disp.', 'Sim 3 Bottom Bracket Y Disp.', 'Sim 3 Bottom Bracket X Rot.', 'Model Mass']:      
        y[col]=[np.abs(val) for val in y[col].values]
        y.rename(columns={col:col+" Magnitude"}, inplace=True)
    return y

def eval_obj(valid, objectives):
    first = True
    for objective in objectives:
        res = objective(valid[:, 0], valid[:, 1])
        if first:
            y = res
            first=False
        else:
            y = np.vstack([y, res])
    y = np.transpose(y)
    return y

def gen_toy_dataset(samplingfunction, validityfunction, objectives, rangearr, scaling):
    valid, invalid = samplingfunction(validityfunction, rangearr)
    if valid == []:
        raise Exception("No Valid Samples Generated")
    #Scale if desired
    if scaling:
        scaler = StandardScaler()
        scaler.fit(valid)
        valid = scaler.transform(valid)
        if not invalid.size==0:
            invalid = scaler.transform(invalid)
    else:
        scaler = None
    return valid, invalid, scaler


def gen_background_plot(validityfunction, rangearr):
    xx, yy = np.mgrid[rangearr[0,0]:rangearr[0,1]:.01, rangearr[1,0]:rangearr[1,1]:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
#     for val in grid:
#         a = val[0]
#         b = val[1]
#         if validityfunction(a,b):
#             Z.append(1)
#         else:
#             Z.append(0)
    a = grid[:,0]
    b = grid[:,1]
    Z = validityfunction(a,b)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    return (xx,yy,Z)

def circles_val_wrapper(rad):
    def circles_val(a, b, rad=rad):
        return np.square(a-np.round(a)) + np.square(b-np.round(b)) >= rad**2
    return circles_val

def ring_val_wrapper(rad, thickness):
    def ring_val(a, b, rad=rad, thickness=thickness):
        rad_act = np.sqrt(np.square(a) + np.square(b)) - rad
        return np.logical_and(np.less_equal(-thickness/2, rad_act), np.less_equal(rad_act, thickness/2))
    return ring_val

def rectangles_val_wrapper(a_sc, b_sc):
    def rectangles_val(a, b, a_sc=a_sc, b_sc=b_sc):
        return (a//a_sc + b//b_sc +1) % 2
    return rectangles_val

def radial_circles_val_wrapper(rad, cr, num):
    def radial_circles_val(a, b, rad=rad, cr=cr, num=num):
        allvalid=np.zeros_like(a)
        for i in range(num):
            angle=i/num*2*np.pi
            x = rad*np.cos(angle)
            y = rad*np.sin(angle)
            dist= np.sqrt(np.square(a-x)+np.square(b-y))
            valid = dist<cr
            allvalid = np.logical_or(valid, allvalid)
        return allvalid
    return radial_circles_val

def concentric_circles_val_wrapper(n_circ):
    def concentric_circles_val(a, b, n_circ=n_circ):
        r = np.sqrt(np.square(a) + np.square(b))
        return np.logical_and((np.floor(r*2*n_circ)%2 ==0), r<((n_circ-0.5)/n_circ))
    return concentric_circles_val

def all_val_wrapper():
    def all_val(a, b):
        return np.ones_like(a)
    return all_val

def circle_obj_wrapper(rad):
    def circle_obj(a, b, rad=rad):
        r = np.sqrt(np.square(a) + np.square(b))
        return 1/(1+np.square(rad - r))
    return circle_obj

def hlines_obj_wrapper(spacing):
    def hlines_obj(a, b, spacing=spacing):
        b=b/spacing
        return 0.25-np.square(b-np.round(b))
    return hlines_obj
def vlines_obj_wrapper(spacing):
    def vlines_obj(a, b, spacing=spacing):
        a=a/spacing
        return 0.25-np.square(a-np.round(a))
    return vlines_obj

def diag_ovals_obj_wrapper(a_sc, b_sc):
    def diag_ovals_obj(a, b, a_sc=a_sc, b_sc=b_sc):
        a = a/a_sc
        b = b/b_sc
        res = 1-np.cos(3*np.pi*(a+b)) + 6*(1-np.cos(0.5*np.pi*(a-b)))
        return res
    return diag_ovals_obj

def vert_valleys_obj_wrapper(a_sc, b_sc):
    def vert_valleys_obj(a, b, a_sc=a_sc, b_sc=b_sc):
        a = a/a_sc
        b = b/b_sc
        res = 0.5 + 0.5*np.sin(3/4*np.pi*b) 
        return res
    return vert_valleys_obj

def diag_ovals_obj_wrapper(a_sc, b_sc):
    def diag_ovals_obj(a, b, a_sc=a_sc, b_sc=b_sc):
        a = a/a_sc
        b = b/b_sc
        res = 1-np.cos(3*np.pi*(a+b)) + 6*(1-np.cos(0.5*np.pi*(a-b)))
        return res
    return diag_ovals_obj

def exp_obj_wrapper(a_sc, b_sc):
    def exp_obj(a, b, a_sc=a_sc, b_sc=b_sc):
        a = a/a_sc
        b = b/b_sc
#         res = 0.5+0.5*np.sin(np.square(a)*np.square(b))
#         res = np.sin(np.square(a))*np.sin(np.square(b))
        p = np.abs(np.multiply(np.power(a, 4), np.power(b, 4)))
        res = 0.15*(np.square(a)+np.square(b))+ 1-np.divide(0.1*(p+1), p+0.1)
        return res
    return exp_obj

def rectangles_obj_wrapper(a_sc, b_sc):
    def rectangles_obj(a, b, a_sc=a_sc, b_sc=b_sc):
        res = (np.sin(a/a_sc*2*np.pi))*(np.sin(b/b_sc*2*np.pi))+1
        return res
    return rectangles_obj

def uniform_obj_wrapper():
    def uniform_obj(a, b):
        return np.ones_like(a)
    return uniform_obj


def sample_uniform_wrapper(psamples, nsamples):
    def sample_uniform(validityfunction, rangearr, psamples=psamples, nsamples=nsamples):
        valid = []
        invalid = []
        while len(valid)<psamples or len(invalid)<nsamples:
            a = random.uniform(*rangearr[0,:])
            b = random.uniform(*rangearr[1,:])
            if validityfunction(a,b):
                if len(valid)<psamples:
                    valid.append([a,b])
            else:
                if len(invalid)<nsamples:
                    invalid.append([a,b])
        valid = np.array(valid)
        invalid = np.array(invalid)
        return valid, invalid
    return sample_uniform

def sample_circle_blobs_wrapper(samples, num_blobs, rad, rs=0.1):
    def sample_circle_blobs(validityfunction, rangearr, samples=samples, num_blobs=num_blobs, rad=rad, rs=rs):
        mode = np.random.randint(0, num_blobs, size = samples)
        angle = mode*np.pi*2/num_blobs
        a = rad*np.cos(angle)
        b = rad*np.sin(angle)
        rs = (np.random.normal(0,1,samples)*rs)
        sa = np.random.uniform(0,2*np.pi, samples)
        a = a + rs*np.cos(sa)
        b = b + rs*np.sin(sa)
        
        validity = validityfunction(a,b)
        allsamples = np.array([a,b]).transpose()
        valid = allsamples[validity.astype(bool)]
        invalid = allsamples[~validity.astype(bool)]
        valid = np.array(valid)
        invalid = np.array(invalid)
        return valid, invalid
    return sample_circle_blobs
# def function_lookup(function):
#     if function== "Two recta":
#     functions.append([rectangles_wrapper(2, 1), np.array([[0,2], [0,2]])])
#     functions.append([concentric_circles_wrapper(1), np.array([[-1,1], [-1,1]])])
#     functions.append([rectangles_wrapper(3, 1), np.array([[0,3], [0,3]])])
#     functions.append([circles_wrapper(0.3), np.array([[0,2], [0,2]])])
#     functions.append([concentric_circles_wrapper(2), np.array([[-1,1], [-1,1]])])
#     functions.append([rectangles_wrapper(1, 1), np.array([[0,3], [0,3]])])
#     functions.append([circles_wrapper(0.3), np.array([[0,4], [0,4]])])
#     functions.append([concentric_circles_wrapper(3), np.array([[-1,1], [-1,1]])])


''' 
    KNO1 Function implementation modified from: 
    Chen, Wei, and Faez Ahmed. "MO-PaDGAN: Reparameterizing Engineering Designs for 
    augmented multi-objective optimization." Applied Soft Computing 113 (2021): 107909.
    https://github.com/wchen459/MO-PaDGAN-Optimization
'''
class Function(object):
    
    def __init__(self):
        pass
    
class NKNO1(Function): 
    '''
    Normalized KNO1
    Reference: 
        J. Knowles. ParEGO: A hybrid algorithm with on-line landscape approximation for 
        expensive multiobjective optimization problems. Technical Report TR-COMPSYSBIO-2004-01, 
        University of Manchester, UK, 2004. Available from http://dbk.ch.umist.ac.uk/knowles/pubs.html
    '''
    def __init__(self):
        self.dim = 2
        self.n_obj = 2
        self.name = 'NKNO1'
        
        x1 = np.linspace(-0.5+4.4116/3-1, 0.5, 100)
        x2 = 4.4116/3 - x1 - 1
        
def calculate_KNO1(a,b):
    a = 3*(a+.5)
    b = 3*(b+.5)
    r = 9 - (3*np.sin(5/2*(a+b)**2) + 3*np.sin(4*(a+b)) + 5*np.sin(2*(a+b)+2))
    phi = np.pi/12*(a-b+3)
    y1 = r/20*np.cos(phi)
    y2 = r/20*np.sin(phi)
    return y1, y2

#Custom KN01 wrapper function
def KNO1_a_wrapper(a_sc, b_sc):
    def KNO1_a(a, b, a_sc=a_sc, b_sc=b_sc):
        a = a/a_sc
        b = b/b_sc
        y1, y2 = calculate_KNO1(a,b)
        return y1
    return KNO1_a

def KNO1_b_wrapper(a_sc, b_sc):
    def KNO1_b(a, b, a_sc=a_sc, b_sc=b_sc):
        a = a/a_sc
        b = b/b_sc
        y1, y2 = calculate_KNO1(a,b)
        return y2
    return KNO1_b
