import numpy as np
import numpyro
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import numpyro.distributions as dist
from numpyro import deterministic

import arviz as az
import jax
import jax.numpy as jnp
from jax import config
from jax.scipy.special import expit


#Main time series model described Section 4.4
def diversity_removals_full(d_mat, num_subs, s_diff_mat, div_diff_mat, r_mat, mask, month_mat):
    #monthly effects
    gamma_std = numpyro.sample("gamma_std", dist.HalfNormal(.5))

    gammas = numpyro.sample("gamma", dist.Normal(0, gamma_std).expand([45]))

    #Effects of change in subscribers on distinctiveness

    #Interaction Effects of removal rate & and growth on distinctiveness
    alpha_r = numpyro.sample("alpha_r", dist.Normal(0,.5))
    #Main effect of removal rate & and growth on distinctiveness
    eta_r = numpyro.sample("eta_r", dist.Normal(0,.5))

    #Interaction Effects of change in diversity & growth on distinctiveness
    alpha_div = numpyro.sample("alpha_div" ,dist.Normal(0,.5))
    eta_div = numpyro.sample("eta_div", dist.Normal(0,.5))

    #intercepts
    alpha_mu = numpyro.sample("alpha_mu", dist.Normal(0,1))
    beta_mu = numpyro.sample("beta_mu", dist.Normal(0, 1))
    sigmas = numpyro.sample("sigmas", dist.Exponential(1).expand([2]))
    L_rho = numpyro.sample("rho", dist.LKJCholesky(2, 2))
    rho = L_rho @ L_rho.T
    cov = jnp.outer(sigmas, sigmas) * rho
    alpha_betas = numpyro.sample(
        "alpha_betas", dist.MultivariateNormal(jnp.stack([alpha_mu, beta_mu]), cov).expand([num_subs])
    )
    print("alpha-betas", type(alpha_betas))
    print("div_diff_mat", type(div_diff_mat))
    print("r_mat", type(r_mat))
    print("alpha_r", type(alpha_r))
    print("alpha_div", type(alpha_div))
    alphas = alpha_betas[:, [0]]  + alpha_r*r_mat + alpha_div*div_diff_mat
    betas = alpha_betas[:, [1]]
    latent_vars = expit(jnp.cumsum((alphas*s_diff_mat + eta_r*r_mat+ eta_div*div_diff_mat + gammas[month_mat]), axis=1) + betas)

    phi = numpyro.sample("phi", dist.Exponential(1))
    with numpyro.handlers.mask(mask=mask):
        distinctiveness = numpyro.sample("d_obs", dist.BetaProportion(latent_vars, phi), obs=d_mat)

def run_model(d_mat, s_differences_mat, r_mat, div_differences_mat, month_mat, mask):
    numpyro.set_host_device_count(4)
    NUTS_inst = NUTS(diversity_removals_full, target_accept_prob=0.7,
                        init_strategy=numpyro.infer.initialization.init_to_mean())
    removal_model = MCMC(NUTS_inst, num_warmup=500, num_samples=500, num_chains=4)
    dat_list = dict(
        d_mat = d_mat,
        num_subs = d_mat.shape[0],
        s_diff_mat = s_differences_mat,
        r_mat= r_mat,
        div_diff_mat = div_differences_mat,
        mask = mask,
        month_mat = month_mat,
    )    
    removal_model.run(jax.random.PRNGKey(23), **dat_list)
    inf_data = az.from_numpyro(removal_model)
    removal_model.print_summary()
    return inf_data
    
def display_hdis(inf_data):
    print("phi", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["phi"]))))
    print("alpha_mu", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["alpha_mu"]))))
    print("alpha_r", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["alpha_r"]))))
    print("alpha_div", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["alpha_div"]))))
    print("eta_r", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["eta_r"]))))
    print("eta_div", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["eta_div"]))))
    print("alpha_std", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["alpha_std"]))))
    print("beta_mu", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["beta_mu"]))))
    print("beta_std", az.hdi(np.ndarray.flatten(np.array(inf_data["posterior"]["beta_std"]))))

    gamma_hdis = az.hdi(np.squeeze(np.array(inf_data["posterior"]["gamma"][0])))
    count = 0
    for year in range(2018, 2022):
        for month in range(1,13):
            #skip first bits where no data
            if not (year == 2018  and month <=5):
                print("{}/{}: {}".format(month, year, gamma_hdis[count]))
                count+=1    
    
def save_global_params(inf_data, loc):
    new_post = {}
    new_post["alpha_mu"] = np.ndarray.flatten(np.array(inf_data["posterior"]["alpha_mu"])).tolist()
    new_post["alpha_r"] = np.ndarray.flatten(np.array(inf_data["posterior"]["alpha_r"])).tolist()
    new_post["alpha_div"] = np.ndarray.flatten(np.array(inf_data["posterior"]["alpha_div"])).tolist()
    new_post["eta_r"] = np.ndarray.flatten(np.array(inf_data["posterior"]["eta_r"])).tolist()
    new_post["eta_div"] = np.ndarray.flatten(np.array(inf_data["posterior"]["eta_div"])).tolist()
    new_post["beta_mu"] = np.ndarray.flatten(np.array(inf_data["posterior"]["beta_mu"])).tolist()    
    with open(loc, "w+") as f:
        f.write(json.dumps(new_post, indent=4))    
    
    
