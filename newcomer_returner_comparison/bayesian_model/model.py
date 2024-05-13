from jax.scipy.special import expit
import arviz as az
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro import deterministic
from jax import config
import numpy as np

config.update("jax_enable_x64", True)
# %config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["stats.hdi_prob"] = 0.95
az.rcParams["stats.ic_scale"] = "deviance"
az.rcParams["stats.information_criterion"] = "waic"

# models the newcomer and returner distinctiveness scores as a function of the subreddit plus growth experienced
# (is higher growth might be associated with lower newcomer distinctiveness?)
def basic_model(outputs_dict, inputs_dict, newcomer_mask, returner_mask, num_subs):
    sub_pop_mu = numpyro.sample("sub_pop_mu", dist.Normal(0, 1).expand([2, 1]))
    sub_pop_sig = numpyro.sample("sub_pop_sig", dist.Exponential(1).expand([2, 1]))
    sub_pop_effects = numpyro.sample("sub_pop_effects", dist.Normal(sub_pop_mu, sub_pop_sig).expand([2, num_subs]))

    sub_diff_effects = numpyro.sample("growth_effect", dist.Normal(0, 1).expand([2, 1]))
    phis = numpyro.sample("phis", dist.Exponential(1).expand([2]))


    latent_n_sup = expit(sub_pop_effects[0, inputs_dict["newcomer_subreddits"]] + sub_diff_effects[0]*inputs_dict["newcomer_sub_diffs"])
    latent_r_sup = expit(sub_pop_effects[1, inputs_dict["returner_subreddits"]] + sub_diff_effects[1]*inputs_dict["returner_sub_diffs"])


    with numpyro.handlers.mask(mask=newcomer_mask):
        pred_n_sup = numpyro.sample("pred_n_sup", dist.BetaProportion(mean=latent_n_sup, concentration=phis[0]), obs=outputs_dict["newcomer_super_scores"])

    with numpyro.handlers.mask(mask=returner_mask):
        pred_r_sup = numpyro.sample("pred_r_sup", dist.BetaProportion(mean=latent_r_sup, concentration=phis[1]), obs=outputs_dict["returner_super_scores"])


def run_model(output_var_dict, predictor_var_dict, mask_dict, num_subs):
    mcmc = MCMC(NUTS(basic_model, target_accept_prob=.8, init_strategy=numpyro.infer.initialization.init_to_sample()), num_warmup=500, num_samples=500, num_chains=4)
    dat_list = {
        "outputs_dict": output_var_dict,
        "inputs_dict": predictor_var_dict,
        "newcomer_mask": mask_dict["newcomer_mask"],
        "returner_mask": mask_dict["returner_mask"],
        "num_subs": num_subs
    }
    mcmc.run(jax.random.PRNGKey(2), **dat_list)
    inf_data = az.from_numpyro(mcmc)
    az.summary(inf_data, var_names=["sub_pop_mu", "sub_pop_sig", "growth_effect", "phis"])
    return inf_data

    
