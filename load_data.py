# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:05:43 2022

@author: Lyle
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
        res = objective(valid)
        if first:
            y = res
            first=False
        else:
            y = np.vstack([y, res])
    y = np.transpose(y)
    return y

def get_dataset_func(samplingfunction, validityfunction, rangearr): 
    def sample(samplingfunction = samplingfunction, validityfunction=validityfunction, rangearr=rangearr):
        distribution, negative = samplingfunction(validityfunction, rangearr)
        return distribution, negative
    return sample

def gen_dataset(datasetfunction, holdout_frac, scaling):
    distribution, negative = datasetfunction()
    # Scale if desired
    if scaling:
        scaler = StandardScaler()
        scaler.fit(distribution)
        distribution = scaler.transform(distribution)
        if not negative.size == 0:
            negative = scaler.transform(negative)
    else:
        scaler = None

    if holdout_frac > 0:
        distribution, holdout = train_test_split(distribution, test_size=holdout_frac, random_state=0)
    else:
        holdout = []

    return distribution, negative, holdout, scaler


def gen_background_plot(validityfunction, rangearr):
    xx, yy = np.mgrid[rangearr[0,0]:rangearr[0,1]:.01, rangearr[1,0]:rangearr[1,1]:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = validityfunction(grid)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    return (xx,yy,Z)

def circles_val_wrapper(rad, scale):
    def circles_val(x, rad=rad, scale=scale):
        a = scale*x[:,0]
        b = scale*x[:,1]
        return np.square(a-np.round(a)) + np.square(b-np.round(b)) >= rad**2
    return circles_val

def ring_val_wrapper(rad, thickness):
    def ring_val(x, rad=rad, thickness=thickness):

        rad_act = np.sqrt(np.square(x[:,0]) + np.square(x[:,1])) - rad
        return np.logical_and(np.less_equal(-thickness/2, rad_act), np.less_equal(rad_act, thickness/2))
    return ring_val

def plus_val_wrapper(l, w):
    def plus_val(x, l=l, w=w):
        a = x[:,0]
        b = x[:,1]
        return np.logical_not(np.logical_or(np.logical_and(np.abs(a)<l/2, np.abs(b)<w/2), np.logical_and(np.abs(a)<w/2, np.abs(b)<l/2)))
    return plus_val

def rectangles_val_wrapper(a_sc, b_sc):
    def rectangles_val(a, b, a_sc=a_sc, b_sc=b_sc):
        a = x[:,0]
        b = x[:,1]
        return (a//a_sc + b//b_sc +1) % 2
    return rectangles_val

def radial_circles_val_wrapper(rad, cr, num):
    def radial_circles_val(x, rad=rad, cr=cr, num=num):
        a = x[:,0]
        b = x[:,1]
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

def inv_radial_circles_val_wrapper(rad, cr, num):
    def radial_circles_val(x, rad=rad, cr=cr, num=num):
        a = x[:,0]
        b = x[:,1]
        allvalid=np.ones_like(a)
        for i in range(num):
            angle=i/num*2*np.pi
            x = rad*np.cos(angle)
            y = rad*np.sin(angle)
            dist= np.sqrt(np.square(a-x)+np.square(b-y))
            valid = dist>cr
            allvalid = np.logical_and(valid, allvalid)
        return allvalid
    return radial_circles_val

def concentric_circles_val_wrapper(n_circ, size):
    def concentric_circles_val(x, n_circ=n_circ, size=size):
        a = x[:,0]
        b = x[:,1]
        r = np.sqrt(np.square(a) + np.square(b))
        return np.logical_and(((r*n_circ - np.floor(r*n_circ))<size), r<1)
    return concentric_circles_val

def all_val_wrapper():
    def all_val(x):
        a = x[:,0]
        return np.ones_like(a)
    return all_val

def circle_obj_wrapper(rad):
    def circle_obj(x, rad=rad):
        a = x[:,0]
        b = x[:,1]
        r = np.sqrt(np.square(a) + np.square(b))
        return 1/(1+np.square(rad - r))
    return circle_obj

def hlines_obj_wrapper(spacing):
    def hlines_obj(x, spacing=spacing):
        a = x[:,0]
        b = x[:,1]
        b=b/spacing
        return 0.25-np.square(b-np.round(b))
    return hlines_obj
def vlines_obj_wrapper(spacing):
    def vlines_obj(x, spacing=spacing):
        a = x[:,0]
        b = x[:,1]
        a=a/spacing
        return 0.25-np.square(a-np.round(a))
    return vlines_obj

def diag_ovals_obj_wrapper(a_sc, b_sc):
    def diag_ovals_obj(x, a_sc=a_sc, b_sc=b_sc):
        a = x[:,0]
        b = x[:,1]
        a = a/a_sc
        b = b/b_sc
        res = 1-np.cos(3*np.pi*(a+b)) + 6*(1-np.cos(0.5*np.pi*(a-b)))
        return res
    return diag_ovals_obj

def vert_valleys_obj_wrapper(a_sc, b_sc):
    def vert_valleys_obj(x, a_sc=a_sc, b_sc=b_sc):
        a = x[:,0]
        b = x[:,1]
        a = a/a_sc
        b = b/b_sc
        res = 0.5 + 0.5*np.sin(3/4*np.pi*b) 
        return res
    return vert_valleys_obj

def diag_ovals_obj_wrapper(a_sc, b_sc):
    def diag_ovals_obj(x, a_sc=a_sc, b_sc=b_sc):
        a = x[:,0]
        b = x[:,1]
        a = a/a_sc
        b = b/b_sc
        res = 1-np.cos(3*np.pi*(a+b)) + 6*(1-np.cos(0.5*np.pi*(a-b)))
        return res
    return diag_ovals_obj

def exp_obj_wrapper(a_sc, b_sc):
    def exp_obj(x, a_sc=a_sc, b_sc=b_sc):
        a = x[:,0]
        b = x[:,1]
        a = a/a_sc
        b = b/b_sc
        p = np.abs(np.multiply(np.power(a, 4), np.power(b, 4)))
        res = 0.15*(np.square(a)+np.square(b))+ 1-np.divide(0.1*(p+1), p+0.1)
        return res
    return exp_obj

def rectangles_obj_wrapper(a_sc, b_sc):
    def rectangles_obj(x, a_sc=a_sc, b_sc=b_sc):
        a = x[:,0]
        b = x[:,1]
        res = (np.sin(a/a_sc*2*np.pi))*(np.sin(b/b_sc*2*np.pi))+1
        return res
    return rectangles_obj

def uniform_obj_wrapper():
    def uniform_obj(x):
        a = x[:,0]
        return np.ones_like(a)
    return uniform_obj


def sample_uniform_wrapper(psamples, nsamples):
    def sample_uniform(validityfunction, rangearr, psamples=psamples, nsamples=nsamples):
        valid = []
        invalid = []
        while len(valid)<psamples or len(invalid)<nsamples:
            samples=10000
            a = np.random.uniform(*rangearr[0,:], size = samples)
            b = np.random.uniform(*rangearr[1,:], size = samples)
            x = np.vstack([a,b]).T
            val = validityfunction(x)
            valididx = np.where(val==1)
            invalidx = np.where(val==0)
            if len(valid)<psamples:
                ls = [[a[i],b[i]] for i in valididx[0]]
                # print(ls)
                valid = valid + ls
            if len(invalid)<nsamples:
                ls = [[a[i],b[i]] for i in invalidx[0]]
                invalid = invalid + ls
        valid = np.array(valid)[:psamples,:]
        invalid = np.array(invalid)[:nsamples,:]
        return valid, invalid
    return sample_uniform

def sample_gaussian_wrapper(psamples, nsamples, r):
    def sample_uniform(validityfunction, rangearr, psamples=psamples, nsamples=nsamples, r=r):
        valid = []
        invalid = []
        while len(valid)<psamples or len(invalid)<nsamples:
            samples=10000
            r_s = (np.random.normal(0,1,samples)*r)
            a_s = np.random.uniform(0,2*np.pi, samples)
            a = r_s*np.cos(a_s)
            b = r_s*np.sin(a_s)
            x = np.vstack([a,b]).T
            val = validityfunction(x)
            valididx = np.where(val==1)
            invalidx = np.where(val==0)
            if len(valid)<psamples:
                ls = [[a[i],b[i]] for i in valididx[0]]
                # print(ls)
                valid = valid + ls
            if len(invalid)<nsamples:
                ls = [[a[i],b[i]] for i in invalidx[0]]
                invalid = invalid + ls
        valid = np.array(valid)[:psamples,:]
        invalid = np.array(invalid)[:nsamples,:]
        return valid, invalid
    return sample_uniform

def sample_circle_blobs_wrapper(psamples, nsamples, num_blobs, rad, rs=0.1):
    def sample_circle_blobs(validityfunction, rangearr, psamples=psamples, nsamples=nsamples, num_blobs=num_blobs, rad=rad, rs=rs):
        valid = []
        invalid = []
        while len(valid)<psamples or len(invalid)<nsamples:
            samples=10000
            mode = np.random.randint(0, num_blobs, size = samples)
            angle = mode*np.pi*2/num_blobs
            a = rad*np.cos(angle)
            b = rad*np.sin(angle)
            rs = (np.random.normal(0,1,samples)*rs)
            sa = np.random.uniform(0,2*np.pi, samples)
            a = a + rs*np.cos(sa)
            b = b + rs*np.sin(sa)
            x = np.vstack([a,b]).T
            val = validityfunction(x)
            valididx = np.where(val==1)
            invalidx = np.where(val==0)
            if len(valid)<psamples:
                ls = [[a[i],b[i]] for i in valididx[0]]
                # print(ls)
                valid = valid + ls
            if len(invalid)<nsamples:
                ls = [[a[i],b[i]] for i in invalidx[0]]
                invalid = invalid + ls
        valid = np.array(valid)[:psamples]
        invalid = np.array(invalid)[:nsamples]
        return valid, invalid
    return sample_circle_blobs



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
        
def calculate_KNO1(a, b):
    a = 3*(a+.5)
    b = 3*(b+.5)
    r = 9 - (3*np.sin(5/2*(a+b)**2) + 3*np.sin(4*(a+b)) + 5*np.sin(2*(a+b)+2))
    phi = np.pi/12*(a-b+3)
    y1 = r/20*np.cos(phi)
    y2 = r/20*np.sin(phi)
    return y1, y2

#Custom KN01 wrapper function
def KNO1_a_wrapper(a_sc, b_sc):
    def KNO1_a(x, a_sc=a_sc, b_sc=b_sc):
        a = x[:,0]
        b = x[:,1]
        a = a/a_sc
        b = b/b_sc
        y1, y2 = calculate_KNO1(a,b)
        return y1
    return KNO1_a

def KNO1_b_wrapper(a_sc, b_sc):
    def KNO1_b(x, a_sc=a_sc, b_sc=b_sc):
        a = x[:,0]
        b = x[:,1]
        a = a/a_sc
        b = b/b_sc
        y1, y2 = calculate_KNO1(a,b)
        return y2
    return KNO1_b
