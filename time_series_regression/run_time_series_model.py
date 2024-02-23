import os
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime as dt
import seaborn as sns
import pandas as pd
import numpyro
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import numpyro.distributions as dist
from numpyro import deterministic
from jax.scipy.special import expit
import arviz as az
import jax
import jax.numpy as jnp
import json
from jax.config import config

from load_data import read_all, convert_to_mat, gen_mask, standardize_and_diff_mat
from model import run_model, display_hdis, save_global_params
config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.95
az.rcParams["stats.ic_scale"] = "deviance"
az.rcParams["stats.information_criterion"] = "waic"

#read in all data sources from file
subs, dists, months, active_users, divs, removals  = read_all("./data/skip_list.txt", "./data/distinctiveness_scores.json",
        "./data/subscriber_info.csv", "./data/active_users_info.csv", "./data/diversity_scores.json", "./data/removal_ratio_info.csv", "data/month_indices.json")

#convert each data source into a matrix. 
#As described in 3.3, for each subreddit, use only the longest contiguous sequence of months with valid data.
# Drop any months were no contiguous sequences (of length >1) of valid data.
#sub_names is created to allow us to map matrix indices back to subreddits
#all time series are -1 padded at the end when there is no data available
sub_names, s_mat, d_mat, r_mat, div_mat, month_mat = convert_to_mat(subs, dists, divs, removals, months)

#mask indicating which sub-month pairs are padding
mask = gen_mask(s_mat, d_mat, r_mat, div_mat)
mean_dict, std_dict = {}, {}

#standardize matrices
r_mat,mean_dict, std_dict = standardize_and_diff_mat(r_mat, mean_dict, std_dict, "r", False)

#last argument set to true since subscribers + diversity scores need to be converted to month-to-month differences
s_differences_mat, mean_dict, std_dict = standardize_and_diff_mat(s_mat, mean_dict, std_dict, "growth", True) 
div_differences_mat, mean_dict, std_dict = standardize_and_diff_mat(div_mat, mean_dict, std_dict, "div", True) 

print("CLEARED PREPROCESSING")

#run the time series model specified in model.py
inf_data = run_model(d_mat, s_differences_mat, r_mat, div_differences_mat, month_mat, mask)

#print out hdi for key model parameters
display_hdis(inf_data)

#Save subset of parameters necessary for posterior plot-making to json
save_global_params(inf_data, "./posterior.json")

with open("./posterior.json", "r") as f:
    post_dict = json.load(f)

#generate triptych plot (Figure 4)
posterior_plot_maker(post_dict, [0, .05], [np.log(i) for i in range(1, 11)], [-0.1, 0, .1], mean_dict, std_dict, .5, loc = "./model_predictions.pdf")


