from matplotlib.ticker import MultipleLocator
import numpy as np
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
import seaborn as sns


def predict_change(posterior_dict, growth, r, div_diff, mean_dict, std_dict, start_point):
    growth_z = (growth - mean_dict["growth"]) / std_dict["growth"]
    div_z = (div_diff - mean_dict["div_diff"]) / std_dict["div_diff"]
    r_z = (r - mean_dict["r"]) / std_dict["r"]
    coef = np.array(posterior_dict["alpha_mu"]) + np.array(posterior_dict["alpha_r"])*r_z + np.array(posterior_dict["alpha_div"])*div_z
    return  np.squeeze(expit(coef*growth_z +r_z*np.array(posterior_dict["eta_r"]) +div_z*np.array(posterior_dict["eta_div"])  + start_point) -expit(start_point))

def posterior_plot_maker(posterior_dict, r_bands, growth_range, diversity_bands, mean_dict, std_dict, start_point, loc):
    num_div = len(diversity_bands)
    colors = ["#fde541", "#4ac6ff", "#ff4a7d"]
    fig, triptych = plt.subplots(2, 3, figsize=(13, 4), sharex=True, sharey=True)
    sns.set(style="white")
    first = True
    for i, r_band in enumerate(r_bands):
        for j, div_diff in enumerate(diversity_bands):
            ax = triptych[i][j]
            ax.set_title("Δ Diversity = {}, Removal rate = {}".format(div_diff, r_band))
            records = []
            xs = [np.exp(val) for val in growth_range]
            ys = [predict_change(posterior_dict, growth, r_band, div_diff, mean_dict, std_dict, start_point) for growth in growth_range]

            for x_ind in range(len(xs)):
                for y_ind in range(len(ys[0])):
                    #print(ys[x_ind][y_ind])
                    records.append({"Growth Factor": xs[x_ind], "Δ Distinctiveness": ys[x_ind][y_ind].item(), "Removal Rate": r_band})
            cur_df = pd.DataFrame.from_records(records)
            sns.despine(left=False, bottom=False)
            g = sns.lineplot(ax=ax, data=cur_df, x="Growth Factor", y="Δ Distinctiveness", errorbar=("pi", 95), color = colors[j])

            majorLocator = MultipleLocator(.01)
            minorLocator = MultipleLocator(.025)
            g.yaxis.set_major_locator(majorLocator)
            g.yaxis.set_minor_locator(minorLocator)

            majorLocator = MultipleLocator(2)
            minorLocator = MultipleLocator(.5)

            g.xaxis.set_major_locator(majorLocator)
            g.xaxis.set_minor_locator(minorLocator)
            g.minorticks_on()
            g.tick_params(direction="out", length=4, width=2, colors="k", which="major", left=True, bottom=True)
            g.tick_params(direction="out", length=2, width=1, colors="k", which="minor", left=True, bottom=True)
            if not first:
                ax.set(xlabel=None)
                ax.set(ylabel=None)
            first = False
    plt.tight_layout()
    plt.savefig(loc)

    plt.show()
    return triptych

