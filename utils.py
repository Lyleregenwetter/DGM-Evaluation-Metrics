import numpy as np
import pandas as pd
import os
import shutil
import matplotlib
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import sklearn 
import time
import pickle
from IPython.display import display

import evaluation
import load_data

def fit_and_generate(functions, methods, numinst, numgen, scaling, obj_status, conditional_status, holdout = 0):
    
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    #Initialize np array to hold generated samples
    generated = np.zeros((len(functions), len(methods), numinst, numgen, 2))
    vs_s = np.zeros((len(functions), numinst)).tolist()
    is_s = np.zeros((len(functions), numinst)).tolist()
    hs_s = np.zeros((len(functions), numinst)).tolist()
    scaler_s = np.zeros((len(functions), numinst)).tolist()
    #Loop over number of model instantiations to test
    for inst in range(numinst):
        #Loop over the problems to test
        for func in range(len(functions)):
            #Unpack various problem parameters
            samplingfunction, validityfunction, objectives, rangearr, cond_func, cond = functions[func]
            
            #Generate the data
            valid_scaled, invalid_scaled, holdout_scaled, scaler = load_data.gen_toy_dataset(samplingfunction, validityfunction, objectives, rangearr, holdout, scaling)
            
            #Get some unscaled versions of the data to use for calculating objectives/condition parameters
            if scaling: 
                valid = scaler.inverse_transform(valid_scaled)
            else:
                valid = valid_scaled
                
            #Evaluate objective values for all datapoints
            if obj_status:
                y_valid = load_data.eval_obj(valid, objectives)
            else:
                y_valid = None
            #Evaluate condition value for all datapoints
            if conditional_status:
                c_valid = load_data.eval_obj(valid, [cond_func])
            else:
                c_valid = None
            #Loop over DGMs
            for i in range(len(methods)): 
                method = methods.values[i]
                
                #Get trained model
                model = method(valid_scaled, invalid_scaled, y_valid, c_valid)
                
                #Call generate function (models are assumed to have a class function, generate)
                x_fake_scaled = model.generate(numgen, np.full(numgen, cond))
                
                #Add to generated results array
                x_fake_scaled = np.array(x_fake_scaled)
                generated[func, i, inst, :, :] = x_fake_scaled
            vs_s[func][inst] = valid_scaled
            is_s[func][inst] = invalid_scaled
            hs_s[func][inst] = holdout_scaled
            scaler_s[func][inst] = scaler
    save_generated(generated, vs_s, is_s, hs_s, scaler_s, timestr)
    return timestr


#Saves generates samples and data
def save_generated(generated, vs_s, is_s, hs_s, scaler_s, timestr):
    #Save and return results
    os.mkdir(f"Results/{timestr}")
    os.mkdir(f"Results/{timestr}/Valid_samples")
    os.mkdir(f"Results/{timestr}/Invalid_samples")
    os.mkdir(f"Results/{timestr}/Holdout_samples")
    os.mkdir(f"Results/{timestr}/Scalers")
    np.save(f"Results/{timestr}/Generated_samples.npy", generated)
    for i in range(len(vs_s)):
        for j in range(len(vs_s[i])):
            np.save(f"Results/{timestr}/Valid_samples/Problem_{i}_Instance_{j}.npy", vs_s[i][j])
            np.save(f"Results/{timestr}/Invalid_samples/Problem_{i}_Instance_{j}.npy", is_s[i][j])
            np.save(f"Results/{timestr}/Holdout_samples/Problem_{i}_Instance_{j}.npy", hs_s[i][j])
            file = open(f"Results/{timestr}/Scalers/Problem_{i}_Instance_{j}.pckl","wb")
            pickle.dump(scaler_s[i][j], file)
#Loads generated samples and data
def load_generated(timestr, numinst, numfunc):
    generated = np.load(f"Results/{timestr}/Generated_samples.npy")
    vs_s = np.zeros((numfunc, numinst)).tolist()
    is_s = np.zeros((numfunc, numinst)).tolist()
    hs_s = np.zeros((numfunc, numinst)).tolist()
    scaler_s = np.zeros((numfunc, numinst)).tolist()
    for i in range(numfunc):
        for j in range(numinst):
            vs_s[i][j] = np.load(f"Results/{timestr}/Valid_samples/Problem_{i}_Instance_{j}.npy")
            is_s[i][j] = np.load(f"Results/{timestr}/Invalid_samples/Problem_{i}_Instance_{j}.npy")
            hs_s[i][j] = np.load(f"Results/{timestr}/Holdout_samples/Problem_{i}_Instance_{j}.npy")
            file = open(f"Results/{timestr}/Scalers/Problem_{i}_Instance_{j}.pckl","rb")
            scaler_s[i][j] = pickle.load(file)
    return generated, vs_s, is_s, hs_s, scaler_s

def plot(ax, rangearr, xx, yy, Z, x, y, x2,y2, title, boundary=0.0, plottype = "generated", validity_status=0, color = "red", target=None):
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}
    plt.rc('font', **font)
    
    black="#000000"
    
    if color == "blue":
        color = "#6C8EBF"
    elif color == "green":
        color = "#82B366"
    elif color == "orange":
        color = "#F2AF00"
    elif color == "yellow":
        color = "#D6B656"
    elif color == "purple": 
        color = "#9673A6"
    elif color == "red": 
        color = "#B85450"
    maincolor = color
        
    white = "#FFFFFF"
#     vgencol = orange
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [white, maincolor], N=7)
    objcmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [white , maincolor], N=6)
    
    s=6
    ax.set_title(title)
    if validity_status==1: #Invalid Samples
        if plottype == "dataset":
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.1)
            ax.scatter(x,y, s=s, c=[maincolor], alpha=0.5)
        elif plottype == "invalid":
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.1)
            ax.scatter(x,y, s=s, c=[black], alpha=0.2)
        elif plottype == "generated":
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.1)
            ax.scatter(x,y, s=s, c=[maincolor])
            ax.scatter(x2,y2, s=s, c=[black])
        elif plottype == "objective":
            img = ax.imshow(Z.T, cmap=objcmap, alpha=0.7, origin='lower', extent = [-2,2,-2,2])    
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            ax.scatter(x,y, s=s, c=black, alpha = 0.3)
            if target:
                CS = ax.contour(Z.T, [target], colors=black, vmin=0, vmax=2, extent = [-2,2,-2,2])
                ax.clabel(CS, fontsize=30, inline=True)
#                 manual_locations = [(-2, 2), (2,-2)]
#                 ax.clabel(CS, fontsize=30, inline=True, manual=manual_locations)
    else:
        if plottype=="objective":
            img = ax.imshow(Z.T, cmap=objcmap, alpha=0.5, origin='lower', extent = [-2,2,-2,2])
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            ax.scatter(x,y, s=s, c="k", alpha = 0.3)
            if target:
                CS = ax.contour(Z.T, [target], colors='k', vmin=0, vmax=2, extent = [-2,2,-2,2])
                ax.clabel(CS, fontsize=30, inline=True)
#                 manual_locations = [(-2, 2), (2,-2)]
#                 ax.clabel(CS, fontsize=30, inline=True, manual=manual_locations)
            
        elif plottype=="dataset": #No invalid Samples
            ax.scatter(x,y, s=s, c=black, alpha=0.15)
            ax.set_title("Original Data")
        elif plottype=="generated":
            ax.scatter(x2,y2, s=s, c=black, alpha=0.05)
            ax.scatter(x,y, s=s, c=maincolor, alpha=0.7)
        elif plottype=="conditional":
            ax.scatter(x2,y2, s=s, c=black, alpha=0.1)
            ax.scatter(x,y, s=s, c=maincolor, alpha=0.7)
    xlen = rangearr[0,1]-rangearr[0,0]
    ylen = rangearr[1,1]-rangearr[1,0]
    ax.set_xlim(rangearr[0,0]-xlen*boundary, rangearr[0,1]+xlen*boundary)
    ax.set_ylim(rangearr[1,0]-ylen*boundary, rangearr[1,1]+ylen*boundary)
    ax.axis('off')
    
def plot_all(timestr, functions, methods, numinst, scaling, validity_status, obj_status, conditional_status, cond_dist, color="red", plotobjs=None):
    generated, vs_s, is_s, hs_s, scaler_s= load_generated(timestr, numinst, len(functions))
    

    #Find the highest number of objectives in any problem
    max_obj = 0
    if obj_status:
        for i in range(len(functions)):
            if len(functions[i][2])>max_obj:
                max_obj = len(functions[i][2])
    
    reset_folder(f"Results/{timestr}/Plots")
    
    #Calculate how many plots per row we will be generating
    plots_in_row = 1 + validity_status +(len(methods)*(1+cond_dist)+max_obj*obj_status + conditional_status+cond_dist)

    #Loop over training instances
    for inst in range(numinst):
        #Initialize subplots
        fig, ax = plt.subplots(len(functions), plots_in_row, figsize=(10*plots_in_row-0.2, 10*len(functions)-0.9))

        #Loop over problems to test
        for func in range(len(functions)):
            #Unpack problem info
            samplingfunction, validityfunction, objectives, rangearr, cond_func, cond = functions[func]
            
            valid_scaled = vs_s[func][inst]
            invalid_scaled = is_s[func][inst]
            scaler = scaler_s[func][inst]

            #Get some unscaled versions for plotting
            if scaling: 
                valid = scaler.inverse_transform(valid_scaled)
                try:
                    invalid = scaler.inverse_transform(invalid_scaled)
                except:
                    invalid = np.array([[None,None]])
            else:
                valid = valid_scaled
                invalid = invalid_scaled

                    
            xx, yy, Z = load_data.gen_background_plot(validityfunction, rangearr)
            plot(fig.axes[plots_in_row*func], rangearr, xx, yy, Z, valid[:,0], valid[:,1], None, None, 
                 "Original valid data", validity_status=validity_status, color=color, plottype = "dataset")
            if validity_status==1:
                plot(fig.axes[plots_in_row*func+1], rangearr, xx, yy, Z, invalid[:,0], invalid[:,1], None, None, 
                     "Original invalid data", validity_status=validity_status, color=color, plottype = "invalid")
            if obj_status:
                num_objectives = len(objectives)
                for i in range(num_objectives):
                    xx_o, yy_o, Z_o = load_data.gen_background_plot(objectives[i], rangearr)
                    obj_idx = plots_in_row*func+1+validity_status+i
                    plot(fig.axes[obj_idx], rangearr, xx_o, yy_o, Z_o, valid[:,0], valid[:,1], None, None, "Objective " +str(i+1), 
                         plottype = "objective", validity_status=validity_status, color=color, target=plotobjs[i])
            else:
                num_objectives = 0
            if conditional_status:
                c_valid = load_data.eval_obj(valid, [cond_func])
                
                xx_o, yy_o, Z_o = load_data.gen_background_plot(cond_func, rangearr)
                cond_idx=plots_in_row*func+1+validity_status+num_objectives*obj_status
                plot(fig.axes[cond_idx], rangearr, xx_o, yy_o, Z_o, valid[:,0], valid[:,1], None, None, "", 
                         plottype = "objective", validity_status=validity_status, color=color, target=plotobjs[num_objectives])
                
            if cond_dist:
                mask = evaluation.get_perc_band(cond, c_valid, 0.1)
                valid_mask = valid[mask]
                if objectives:
                    y_valid_mask = y_valid[mask]
                else:
                    y_valid_mask = None
                valid_scaled_mask = valid_scaled[mask]
                cond_mask = c_valid[mask]

            #Loop over methods to test
            for i in range(len(methods)): 
                if cond_dist:
                    plot(fig.axes[cond_idx+1], rangearr, xx, yy, Z, valid_mask[:,0], valid_mask[:,1], None, None,
                             methods.index[i], validity_status=validity_status, color=color, plottype = "dataset")
                    
                x_fake_scaled = generated[func, i, inst, :, :]
                
                if scaling==True:
                    x_fake = scaler.inverse_transform(x_fake_scaled)
                else:
                    x_fake = x_fake_scaled
                    
                if objectives:
                    y_fake = load_data.eval_obj(x_fake, objectives)
                else:
                    y_fake = None
                
                res_idx=plots_in_row*func+1+validity_status+max_obj*obj_status+i*(1+cond_dist)+conditional_status+cond_dist
                if validity_status:
                    labels, _ = evaluation.evaluate_validity(x_fake, validityfunction)
                    labels = labels.astype(bool)
                    plot(fig.axes[res_idx], rangearr, xx, yy, Z, x_fake[:,0][labels], x_fake[:,1][labels], x_fake[:,0][~labels], x_fake[:,1][~labels], 
                         methods.index[i], plottype="generated", validity_status = validity_status)                
                else:
                    plot(fig.axes[res_idx], rangearr, xx, yy, Z, x_fake[:,0], x_fake[:,1], valid[:,0], valid[:,1], methods.index[i], 
                         plottype="generated", validity_status = validity_status, color=color)
                if cond_dist:
                    plot(fig.axes[res_idx+1], rangearr, xx, yy, Z, x_fake[:,0], x_fake[:,1], valid_mask[:,0], valid_mask[:,1],
                         methods.index[i], validity_status=validity_status, color=color, plottype = "conditional")
        plt.show()
        
        
        fig.savefig(f"Results/{timestr}/Plots/Instance_{inst}.png", dpi=400, transparent=False, facecolor='w')
        
    if numinst>1:
        generate_anim(timestr)
        
def generate_anim(timestr):
    frames=[]
    for filename in os.listdir(f"Results/{timestr}/Plots/"):
        frames.append(imageio.imread(f"Results/{timestr}/Plots/{filename}"))
    imageio.mimsave(f"Results/{timestr}/Allplots.gif", frames, 'GIF', fps=1)        
    
    
def score(timestr, functions, methods, metrics, numinst, scaling, cond_dist, scorebars, plotobjs=None, style=True):
    generated, vs_s, is_s, hs_s, scaler_s= load_generated(timestr, numinst, len(functions))
    
    scores=np.zeros((len(functions), len(methods), len(metrics), numinst))

    #Loop over training instances
    for inst in range(numinst):

        #Loop over problems to test
        for func in range(len(functions)):
            #Unpack problem info
            samplingfunction, validityfunction, objectives, rangearr, cond_func, cond = functions[func]
            
            valid_scaled = vs_s[func][inst]
            invalid_scaled = is_s[func][inst]
            holdout_scaled = hs_s[func][inst]
            scaler = scaler_s[func][inst]

            #Get some unscaled versions for plotting
            if scaling: 
                valid = scaler.inverse_transform(valid_scaled)
                try:
                    invalid = scaler.inverse_transform(invalid_scaled)
                except:
                    invalid = np.array([[None,None]])
                try:
                    holdout = scaler.inverse_transform(holdout_scaled)
                except:
                    holdout = np.array([[None,None]])
            else:
                valid = valid_scaled
                invalid = invalid_scaled
                holdout = holdout_scaled
                
            #Evaluate objective values for all datapoints
            if objectives:
                y_valid = load_data.eval_obj(valid, objectives)
                
                #If rediscovery in metrics, calculate y values for holdout
                if "Rediscovery" in metrics:
                    holdout_y = load_data.eval_obj(valid, objectives)
            else:
                y_valid=None
                
            if cond_dist:
                c_valid = load_data.eval_obj(valid, [cond_func])
                mask = evaluation.get_perc_band(cond, c_valid, 0.1)
                valid_mask = valid[mask]
                if objectives:
                    y_valid_mask = y_valid[mask]
                else:
                    y_valid_mask = None
                valid_scaled_mask = valid_scaled[mask]

            #Loop over methods to test
            for i in range(len(methods)): 
                    
                x_fake_scaled = generated[func, i, inst, :, :]
                
                if scaling==True:
                    x_fake = scaler.inverse_transform(x_fake_scaled)
                else:
                    x_fake = x_fake_scaled
                    
                if objectives:
                    y_fake = load_data.eval_obj(x_fake, objectives)
                else:
                    y_fake = None
                
                for j in range(len(metrics)):
                    if metrics.values[j][1]=="Validity":
                        allscores, meanscore = evaluation.evaluate_validity(x_fake, validityfunction)
                    elif metrics.values[j][1]=="Rediscovery":
                        allscores, meanscore = metrics.values[j][2](x_fake_scaled, y_fake, holdout, holdout_y, invalid_scaled, scorebars)
                    elif metrics.values[j][1]=="Conditioning Reconstruction":
                        allscores, meanscore = metrics.values[j][2](x_fake_scaled, y_fake, valid_scaled, c_valid, invalid_scaled, scorebars)
                    elif metrics.values[j][1]=="Conditioning Adherence":
                        c_gen = load_data.eval_obj(x_fake, [cond_func])
                        allscores=None
                        meanscore = sklearn.metrics.mean_squared_error(c_gen, np.ones_like(c_gen)*cond)
                    else:
                        if cond_dist:
                            allscores, meanscore = metrics.values[j][1](x_fake_scaled, y_fake, valid_scaled_mask, y_valid_mask, invalid_scaled, scorebars)
                        else:
                            allscores, meanscore = metrics.values[j][1](x_fake_scaled, y_fake, valid_scaled, y_valid, invalid_scaled, scorebars)
                    scores[func, i, j, inst] = meanscore

    reset_folder(f"Results/{timestr}/Scores")
    
    for i in range(np.shape(scores)[0]):
        meanscores = np.mean(scores[i], axis=(2))
        stds = np.std(scores[i], axis=(2))
        scoredf_raw = pd.DataFrame(meanscores, index=methods.index, columns = metrics.index).transpose()
        if numinst>1:
            scoredf = append_error(scoredf_raw, stds)
        if style:
            scoredf = highlight_best(scoredf, scoredf_raw, [v[0] for v in metrics.values])
        scoredf.columns.name=f"Problem {i+1} Scores:"
        scoredf.to_excel(f"Results/{timestr}/Scores/problem_{i+1}_scores.xlsx", index_label=scoredf.columns.name)
        if not style:
            scoredf.to_csv(f"Results/{timestr}/Scores/problem_{i+1}_scores.csv", index_label=scoredf.columns.name)
        display(scoredf)

    #average scores
    meanscores = np.mean(scores, axis=(0,3))
    stds = np.std(scores, axis=(0,3))
    scoredf_raw = pd.DataFrame(meanscores, index=methods.index, columns = metrics.index).transpose()
    if numinst*len(functions)>1:
        scoredf = append_error(scoredf_raw, stds)
    if style:
        scoredf = highlight_best(scoredf, scoredf_raw, [v[0] for v in metrics.values])
    scoredf.columns.name = "Average scores:"
    scoredf.to_excel(f"Results/{timestr}/Scores/average_scores.xlsx", index_label=scoredf.columns.name)
    if not style:
        scoredf.to_csv(f"Results/{timestr}/Scores/average_scores.csv", index_label=scoredf.columns.name)
    display(scoredf)

def reset_folder(folder):     
    if os.path.exists(folder) and os.path.isdir(folder):
        shutil.rmtree(folder)                
    os.mkdir(folder)
        
def highlight_bold(s, df_ref, directions):
    v = df_ref.values
    max_mask = (v == np.amax(v, keepdims=True, axis=1))
    min_mask = (v == np.amin(v, keepdims=True, axis=1))
    
    dir_bool = np.array([direction=="maximize" for direction in directions])[np.newaxis].transpose()
    mask = (dir_bool*max_mask + (1-dir_bool)*min_mask).astype("bool")
    mask = pd.DataFrame(mask, index = df_ref.index, columns=df_ref.columns)
    mask.replace(True, "font-weight: bold", inplace=True)
    mask.replace(False, None, inplace=True)
    return mask.values

    
def highlight_best(df, df_ref, directions):
    df = df.style.apply(highlight_bold, axis=None, df_ref=df_ref, directions=directions)
    return df

def append_error(df, stds):
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            df.iloc[i,j] = f"{df.iloc[i,j]:.3f}\u00B1{stds[j,i]:.3f}"
    return df