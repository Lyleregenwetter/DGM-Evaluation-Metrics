import numpy as np
import pandas as pd
import os
import glob
import shutil
import matplotlib
import itertools as it
from matplotlib import pyplot as plt
from matplotlib import gridspec
from abc import ABCMeta, abstractmethod
import seaborn as sns
import imageio
import sklearn 
import time
import pickle
import textwrap
from IPython.display import display

import evaluation
import load_data

def fit_and_generate(functions, methods, numinst, numanim, numgen, scaling, obj_status, conditional_status, holdout = 0):
    
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    #Initialize np array to hold generated samples
    generated = np.zeros((len(functions), len(methods), numinst, numanim, numgen, 2))
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
                
            checkpoint_steps=[]
            #Loop over DGMs
            for i in range(len(methods)): 
                method = methods.values[i]
                
                #Train and return generated samples
                x_fake_scaled, cps = method(valid_scaled, invalid_scaled, y_valid, c_valid, numgen, numanim, np.full(numgen, np.full(numgen, cond)))
                checkpoint_steps.append(cps)
                
                #Add to generated results array
                x_fake_scaled = np.array(x_fake_scaled)
                generated[func, i, inst, :, :, :] = x_fake_scaled
            vs_s[func][inst] = valid_scaled
            is_s[func][inst] = invalid_scaled
            hs_s[func][inst] = holdout_scaled
            scaler_s[func][inst] = scaler
    checkpoint_steps = np.stack(checkpoint_steps)
    save_generated(generated, checkpoint_steps, vs_s, is_s, hs_s, scaler_s, timestr)
    return timestr


#Saves generates samples and data
def save_generated(generated, checkpoint_steps, vs_s, is_s, hs_s, scaler_s, timestr):
    #Save and return results
    os.mkdir(f"Results/{timestr}")
    os.mkdir(f"Results/{timestr}/Valid_samples")
    os.mkdir(f"Results/{timestr}/Checkpoint_steps")
    os.mkdir(f"Results/{timestr}/Invalid_samples")
    os.mkdir(f"Results/{timestr}/Holdout_samples")
    os.mkdir(f"Results/{timestr}/Scalers")
    np.save(f"Results/{timestr}/Generated_samples.npy", generated)
    np.save(f"Results/{timestr}/Checkpoint_steps.npy", checkpoint_steps)
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
    checkpoint_steps = np.load(f"Results/{timestr}/Checkpoint_steps.npy")
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
    return generated, checkpoint_steps, vs_s, is_s, hs_s, scaler_s

#Generates a single 2D data plot
def plot(ax, rangearr, xx, yy, Z, x, y, x2,y2, title, boundary=0.0, plottype = "generated", validity_status=0, color = "red", target=None):
    
    font = {'weight' : 'bold',
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
    generated, checkpoint_steps, vs_s, is_s, hs_s, scaler_s= load_generated(timestr, numinst, len(functions))
    steps = np.shape(generated)[3]
    #Find the highest number of objectives in any problem
    max_obj = 0
    if obj_status:
        for i in range(len(functions)):
            if len(functions[i][2])>max_obj:
                max_obj = len(functions[i][2])
    
    reset_folder(f"Results/{timestr}/Plots")
    reset_folder(f"Results/{timestr}/Animations")
    #Calculate how many plots per row we will be generating
    plots_in_row = 1 + validity_status +(len(methods)*(1+cond_dist)+max_obj*obj_status + conditional_status+cond_dist)

    #Loop over training instances
    for inst in range(numinst):
        for step in range(steps):
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
                    epoch = checkpoint_steps[i, step]
                    if cond_dist:
                        plot(fig.axes[cond_idx+1], rangearr, xx, yy, Z, valid_mask[:,0], valid_mask[:,1], None, None,
                                 methods.index[i], validity_status=validity_status, color=color, plottype = "dataset")
                    #TODO Make anim
                    x_fake_scaled = generated[func, i, inst, step, :, :]

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
                        plot(fig.axes[res_idx], rangearr, xx, yy, Z, x_fake[:,0][labels], x_fake[:,1][labels], x_fake[:,0][~labels], x_fake[:,1][~labels], f"{methods.index[i]}: Epoch {epoch}", plottype="generated", validity_status = validity_status, color=color)                
                    else:
                        plot(fig.axes[res_idx], rangearr, xx, yy, Z, x_fake[:,0], x_fake[:,1], valid[:,0], valid[:,1], f"{methods.index[i]}: Epoch {epoch}", 
                             plottype="generated", validity_status = validity_status, color=color)
                    if cond_dist:
                        plot(fig.axes[res_idx+1], rangearr, xx, yy, Z, x_fake[:,0], x_fake[:,1], valid_mask[:,0], valid_mask[:,1],
                             f"{methods.index[i]}: Epoch {epoch}", validity_status=validity_status, color=color, plottype = "conditional")
            if step==steps-1: #Display plot on the last step
                plt.show()
            plt.close(fig)
            
            fig.savefig(f"Results/{timestr}/Plots/Instance_{inst}_{step}.png", dpi=150, transparent=False, facecolor='w')
        
        
        #Generate training animations for each instance
        if steps>1:
            generate_training_anim(timestr, checkpoint_steps, inst, steps)
    
        
    
    
    if numinst>1:
        generate_final_anim(timestr, numinst, step)
        
def generate_training_anim(timestr, checkpoint_steps, inst, steps):
    frames=[]
    for step in range(steps):
        frames.append(imageio.imread(f"Results/{timestr}/Plots/Instance_{inst}_{step}.png"))
    imageio.mimsave(f"Results/{timestr}/Animations/Instance_{inst}.gif", frames, 'GIF', fps=1)   
        
def generate_final_anim(timestr, numinst, step):
    frames=[]
    for inst in range(numinst):
        frames.append(imageio.imread(f"Results/{timestr}/Plots/Instance_{inst}_{step}.png"))
    imageio.mimsave(f"Results/{timestr}/Animations/All_final_gen.gif", frames, 'GIF', fps=1)        
    
    
def score(timestr, functions, methods, metrics, numinst, scaling, cond_dist, scorebars, plotobjs=None, style=True, plotscores=False):
    generated, checkpoint_steps, vs_s, is_s, hs_s, scaler_s= load_generated(timestr, numinst, len(functions))
    steps = np.shape(generated)[3]
    scores_steps=np.zeros((len(functions), len(methods), len(metrics), numinst, steps))

    #Loop over training instances
    for inst in range(numinst):
        for step in range(steps):
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
                    x_fake_scaled = generated[func, i, inst, step, :, :]
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
                        scores_steps[func, i, j, inst, step] = meanscore
                    clear_temporary_files()
    scores = scores_steps[:,:,:,:,-1] #Isolate only the final scores (no intermediate training scores)
    
    reset_folder(f"Results/{timestr}/Scores")
    if plotscores:
        colors = ["#DC267F", "#648FFF", "#42b27a", "#785EF0", "#FFB000", "#FE6100"]
        barplot(scores, methods.index, metrics.index, [v[0] for v in metrics.values], timestr, colors)
        trainingplots(scores_steps, checkpoint_steps, methods.index, metrics.index, [v[0] for v in metrics.values], timestr, colors)
                      
    for i in range(np.shape(scores)[0]):
        meanscores = np.mean(scores[i], axis=(2))
        stds = np.std(scores[i], axis=(2))
        scoredf_raw = pd.DataFrame(meanscores, index=methods.index, columns = metrics.index).transpose()
        if numinst>1:
            scoredf = append_error(scoredf_raw, stds)
        else:
            scoredf = scoredf_raw.copy()
        if style:
            scoredf = highlight_best(scoredf, scoredf_raw, [v[0] for v in metrics.values])
        scoredf.columns.name=f"Problem {i+1} Scores:"
        scoredf.to_excel(f"Results/{timestr}/Scores/problem_{i+1}_scores.xlsx", index_label=scoredf.columns.name)
        if not style:
            scoredf.to_csv(f"Results/{timestr}/Scores/problem_{i+1}_scores.csv", index_label=scoredf.columns.name)
        display(scoredf)

    #average scores
    if len(functions)>1:
        meanscores = np.mean(scores, axis=(0,3))
        stds = np.std(scores, axis=(0,3))
        scoredf_raw = pd.DataFrame(meanscores, index=methods.index, columns = metrics.index).transpose()
        scoredf = append_error(scoredf_raw, stds)
        if style:
            scoredf = highlight_best(scoredf, scoredf_raw, [v[0] for v in metrics.values])
        scoredf.columns.name = "Average scores:"
        scoredf.to_excel(f"Results/{timestr}/Scores/average_scores.xlsx", index_label=scoredf.columns.name)
        if not style:
            scoredf.to_csv(f"Results/{timestr}/Scores/average_scores.csv", index_label=scoredf.columns.name)
        display(scoredf)

def clear_temporary_files():
    for filename in glob.glob("temp_eval_*"):
        os.remove(filename) 
        
def trainingplots(scores_steps, checkpoint_steps, modelnames, metricnames, directions, timestr, colors):
    numfunctions, nummethods, nummetrics, numinst, numsteps = np.shape(scores_steps)
    metricnames = [f"{metricnames[i]} ({directions[i]})" for i in range(len(directions))]
    font = {'weight' : 'normal', 'size'   : 12}
    plt.rc('font', **font)
    fig, ax = plt.subplots(nummethods, nummetrics, figsize=(4*nummetrics+0.5, 3*numfunctions+0.5))
    if len(np.shape(ax))==1:
        ax = np.expand_dims(ax, axis=0)
    for method in range(nummethods):
        for metric in range(nummetrics):
            alldfs = []
            for inst in range(numinst):
                scores = scores_steps[:, method, metric, inst, :]
                columns = [f"{i}" for i in range(numfunctions)]
                df = pd.DataFrame(scores.T, columns = columns, index = checkpoint_steps[method])
#                 print(df)
                df = pd.melt(df, ignore_index=False, var_name = "Problem", value_name = "Score")
                df.reset_index(inplace=True)
                df = df.rename(columns = {'index':'Step'})
                alldfs.append(df)
            df = pd.concat(alldfs)
            df.index = range(len(df.index))
            axis = ax[method, metric]
            palette = colors*int(np.ceil(len(colors)/numfunctions)) #repeat colors if too few provided
            sns.lineplot(ax = axis, data = df, x = "Step", y = "Score", hue = "Problem", palette=palette[:numfunctions])
            axis.set_title(metricnames[metric], fontsize=12)
            axis.set_ylabel(modelnames[method])
    fig.tight_layout()
    plt.show()
            
        
def barplot(scores, modelnames, metricnames, directions, timestr, colors):
    dfs=[]
    numfunctions, nummethods, nummetrics, numinst = np.shape(scores)
    width = 0.3*numfunctions*(1+nummethods)+2
    arrangement = SquareStrategy().get_grid_arrangement(nummetrics)
    grid = SquareStrategy().get_grid(nummetrics)
    fig = plt.figure(figsize=(max(arrangement)*width, 5*len(arrangement)))
    metricnames = [f"{metricnames[i]} ({directions[i]})" for i in range(len(directions))]
    count=0
    font = {'weight' : 'normal', 'size'   : 12}
    plt.rc('font', **font)
    for metric, sub in enumerate(grid):
        ax = fig.add_subplot(sub)
        for inst in range(numinst):
            columns = [f"Problem {i}" for i in range(numfunctions)]
            df = pd.DataFrame(scores[:,:,metric,inst].T, columns = columns, index = modelnames)
            df = pd.melt(df, ignore_index=False, var_name = "Problem", value_name = "Score")
            df["Model"] = df.index
            dfs.append(df)
        all_df = pd.concat(dfs, axis=0)
        all_df = all_df.reset_index()
        
        g = sns.barplot(ax = ax, data = all_df, x="Problem", y = "Score", hue = "Model", palette = colors)
        plt.title(textwrap.fill(metricnames[metric], width=20, break_long_words=True))
        ax = plt.gca()
        ax.set(xlabel=None)
#         for p in ax.patches: #Align text on bars 
#             txt = p.get_height()
#             txt_x = p.get_x() + p.get_width() / 2
#             value = '{:.2f}'.format(p.get_height())

#         #Wrap labels
#         labels = []
#         for label in ax.get_xticklabels(): 
#             text = label.get_text()
#             labels.append(textwrap.fill(text, width=15, break_long_words=True))
#         ax.set_xticklabels(labels, rotation=0)
    fig.tight_layout()
    plt.show()
    plt.savefig(f"Results/{timestr}/Scores/problem_{metricnames[metric]}_scores.png")

    
    
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


#The following taken from https://github.com/matplotlib/grid-strategy
class GridStrategy(metaclass=ABCMeta):
    """
    Static class used to compute grid arrangements given the number of subplots
    you want to show. By default, it goes for a symmetrical arrangement that is
    nearly square (nearly equal in both dimensions).
    """

    def __init__(self, alignment="center"):
        self.alignment = alignment

    def get_grid(self, n):
        """  
        Return a list of axes designed according to the strategy.
        Grid arrangements are tuples with the same length as the number of rows,
        and each element specifies the number of colums in the row.
        Ex (2, 3, 2) leads to the shape
             x x 
            x x x
             x x
        where each x would be a subplot.
        """

        grid_arrangement = self.get_grid_arrangement(n)
        return self.get_gridspec(grid_arrangement)

    @classmethod
    @abstractmethod
    def get_grid_arrangement(cls, n):  # pragma: nocover
        pass

    def get_gridspec(self, grid_arrangement):
        nrows = len(grid_arrangement)
        ncols = max(grid_arrangement)

        # If it has justified alignment, will not be the same as the other alignments
        if self.alignment == "justified":
            return self._justified(nrows, grid_arrangement)
        else:
            return self._ragged(nrows, ncols, grid_arrangement)

    def _justified(self, nrows, grid_arrangement):
        ax_specs = []
        num_small_cols = np.lcm.reduce(grid_arrangement)
        gs = gridspec.GridSpec(
            nrows, num_small_cols, figure=plt.figure(constrained_layout=True)
        )
        for r, row_cols in enumerate(grid_arrangement):
            skip = num_small_cols // row_cols
            for col in range(row_cols):
                s = col * skip
                e = s + skip

                ax_specs.append(gs[r, s:e])
        return ax_specs

    def _ragged(self, nrows, ncols, grid_arrangement):
        if len(set(grid_arrangement)) > 1:
            col_width = 2
        else:
            col_width = 1

        gs = gridspec.GridSpec(
            nrows, ncols * col_width, figure=plt.figure(constrained_layout=True)
        )

        ax_specs = []
        for r, row_cols in enumerate(grid_arrangement):
            # This is the number of missing columns in this row. If some rows
            # are a different width than others, the column width is 2 so every
            # column skipped at the beginning is also a missing slot at the end.
            if self.alignment == "left":
                # This is left-justified (or possibly full justification)
                # so no need to skip anything
                skip = 0
            elif self.alignment == "right":
                # Skip two slots for every missing plot - right justified.
                skip = (ncols - row_cols) * 2
            else:
                # Defaults to centered, as that is the default value for the class.
                # Skip one for each missing column - centered
                skip = ncols - row_cols

            for col in range(row_cols):
                s = skip + col * col_width
                e = s + col_width

                ax_specs.append(gs[r, s:e])

        return ax_specs


class SquareStrategy(GridStrategy):
    SPECIAL_CASES = {3: (2, 1), 5: (2, 3)}

    @classmethod
    def get_grid_arrangement(cls, n):
        """
        Return an arrangement of rows containing ``n`` axes that is as close to
        square as looks good.
        :param n:
            The number of plots in the subplot
        :return:
            Returns a  :class:`tuple` of length ``nrows``, where each element
            represents the number of plots in that row, so for example a 3 x 2
            grid would be represented as ``(3, 3)``, because there are 2 rows
            of length 3.
        **Example:**
        .. code::
            >>> GridStrategy.get_grid(7)
            (2, 3, 2)
            >>> GridStrategy.get_grid(6)
            (3, 3)
        """
        if n in cls.SPECIAL_CASES:
            return cls.SPECIAL_CASES[n]

        # May not work for very large n
        n_sqrtf = np.sqrt(n)
        n_sqrt = int(np.ceil(n_sqrtf))

        if n_sqrtf == n_sqrt:
            # Perfect square, we're done
            x, y = n_sqrt, n_sqrt
        elif n <= n_sqrt * (n_sqrt - 1):
            # An n_sqrt x n_sqrt - 1 grid is close enough to look pretty
            # square, so if n is less than that value, will use that rather
            # than jumping all the way to a square grid.
            x, y = n_sqrt, n_sqrt - 1
        elif not (n_sqrt % 2) and n % 2:
            # If the square root is even and the number of axes is odd, in
            # order to keep the arrangement horizontally symmetrical, using a
            # grid of size (n_sqrt + 1 x n_sqrt - 1) looks best and guarantees
            # symmetry.
            x, y = (n_sqrt + 1, n_sqrt - 1)
        else:
            # It's not a perfect square, but a square grid is best
            x, y = n_sqrt, n_sqrt

        if n == x * y:
            # There are no deficient rows, so we can just return from here
            return tuple(x for i in range(y))

        # If exactly one of these is odd, make it the rows
        if (x % 2) != (y % 2) and (x % 2):
            x, y = y, x

        return cls.arrange_rows(n, x, y)

    @classmethod
    def arrange_rows(cls, n, x, y):
        """
        Given a grid of size (``x`` x ``y``) to be filled with ``n`` plots,
        this arranges them as desired.
        :param n:
            The number of plots in the subplot.
        :param x:
            The number of columns in the grid.
        :param y:
            The number of rows in the grid.
        :return:
            Returns a :class:`tuple` containing a grid arrangement, see
            :func:`get_grid` for details.
        """
        part_rows = (x * y) - n
        full_rows = y - part_rows

        f = (full_rows, x)
        p = (part_rows, x - 1)

        # Determine which is the more and less frequent value
        if full_rows >= part_rows:
            size_order = f, p
        else:
            size_order = p, f

        # ((n_more, more_val), (n_less, less_val)) = size_order
        args = it.chain.from_iterable(size_order)

        if y % 2:
            return cls.stripe_odd(*args)
        else:
            return cls.stripe_even(*args)

    @classmethod
    def stripe_odd(cls, n_more, more_val, n_less, less_val):
        """
        Prepare striping for an odd number of rows.
        :param n_more:
            The number of rows with the value that there's more of
        :param more_val:
            The value that there's more of
        :param n_less:
            The number of rows that there's less of
        :param less_val:
            The value that there's less of
        :return:
            Returns a :class:`tuple` of striped values with appropriate buffer.
        """
        (n_m, m_v) = n_more, more_val
        (n_l, l_v) = n_less, less_val

        # Calculate how much "buffer" we need.
        # Example (b = buffer number, o = outer stripe, i = inner stripe)
        #    4, 4, 5, 4, 4 -> b, o, i, o, b  (buffer = 1)
        #    4, 5, 4, 5, 4 -> o, i, o, i, o  (buffer = 0)
        n_inner_stripes = n_l
        n_buffer = (n_m + n_l) - (2 * n_inner_stripes + 1)
        assert n_buffer % 2 == 0, (n_more, n_less, n_buffer)
        n_buffer //= 2

        buff_tuple = (m_v,) * n_buffer
        stripe_tuple = (m_v, l_v) * n_inner_stripes + (m_v,)

        return buff_tuple + stripe_tuple + buff_tuple

    @classmethod
    def stripe_even(cls, n_more, more_val, n_less, less_val):
        """
        Prepare striping for an even number of rows.
        :param n_more:
            The number of rows with the value that there's more of
        :param more_val:
            The value that there's more of
        :param n_less:
            The number of rows that there's less of
        :param less_val:
            The value that there's less of
        :return:
            Returns a :class:`tuple` of striped values with appropriate buffer.
        """
        total = n_more + n_less
        if total % 2:
            msg = "Expected an even number of values, got {} + {}".format(
                n_more, n_less
            )
            raise ValueError(msg)

        assert n_more >= n_less, (n_more, n_less)

        # See what the minimum unit cell is
        n_l_c, n_m_c = n_less, n_more
        num_div = 0
        while True:
            n_l_c, lr = divmod(n_l_c, 2)
            n_m_c, mr = divmod(n_m_c, 2)
            if lr or mr:
                break

            num_div += 1

        # Maximum number of times we can half this to get a "unit cell"
        n_cells = 2 ** num_div

        # Make the largest possible odd unit cell
        cell_s = total // n_cells  # Size of a unit cell

        cell_buff = int(cell_s % 2 == 0)  # Buffer is either 1 or 0
        cell_s -= cell_buff
        cell_nl = n_less // n_cells
        cell_nm = cell_s - cell_nl

        if cell_nm == 0:
            stripe_cell = (less_val,)
        else:
            stripe_cell = cls.stripe_odd(cell_nm, more_val, cell_nl, less_val)

        unit_cell = (more_val,) * cell_buff + stripe_cell

        if num_div == 0:
            return unit_cell

        stripe_out = unit_cell * (n_cells // 2)
        return tuple(reversed(stripe_out)) + stripe_out