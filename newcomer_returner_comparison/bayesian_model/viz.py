import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from jax.scipy.special import expit

def viz_variance(ax, indices, indices_sample, posterior, newcomer_subreddits, newcomer_sub_diffs, newcomer_dists, color, mean_color, label, title):
    subreddits = newcomer_subreddits[indices]
    sub_diffs = newcomer_sub_diffs[indices]
    dists = newcomer_dists[indices]
    print(posterior.shape["sub_pop_effects"]) 
    diff = expit(posterior["sub_pop_effects"][:, 0, subreddits] + posterior["growth_effect"][:, 0]*sub_diffs) -\
      expit(posterior["sub_pop_effects"][:, 1, subreddits] + posterior["growth_effect"][:, 1]*sub_diffs)
    mean_diff = np.mean(diff, axis=-1)
    print(az.hdi(np.array(mean_diff)))

    subreddits_sample = newcomer_subreddits[indices_sample]
    for sub in subreddits_sample:
        print(id2sub[sub])
    print("~~~~~~~~~~~~~~~~~~~~~")
    sub_diffs_sample = newcomer_sub_diffs[indices_sample]
    dists_sample = newcomer_dists[indices_sample]
    diff = expit(posterior["sub_pop_effects"][:, 0, subreddits_sample] + posterior["growth_effect"][:, 0]*sub_diffs_sample) -\
      expit(posterior["sub_pop_effects"][:, 1, subreddits_sample] + posterior["growth_effect"][:, 1]*sub_diffs_sample)

    records = []
    mean_records = []
    for iter in range(0, diff.shape[0]):
        for sub in range(0, diff.shape[1]):
            records.append({"iter": iter, "sub": sub, "Returner-Newcomer Gap": diff[iter, sub].item()})
            mean_records.append({"iter": iter, "sub": sub, "mean":  mean_diff[iter].item()})
    cur_df = pd.DataFrame.from_records(records)
    mean_df = pd.DataFrame.from_records(mean_records)
    g = sns.pointplot(ax=ax, data=cur_df, x="sub", y="Returner-Newcomer Gap", errorbar=('pi', 95), join=False, color=color)
    g2 = sns.lineplot(ax=ax, data=mean_df, x="sub", y="mean", errorbar=("pi", 95), err_style='band', color=mean_color, alpha=.25)
    #modify ticks
    majorLocator = MultipleLocator(.04)
    minorLocator = MultipleLocator(.02)
    g.yaxis.set_major_locator(majorLocator)
    g.yaxis.set_minor_locator(minorLocator)
    g.minorticks_on()
    g.tick_params(direction="out", length=4, width=2, colors="k", which="major", left=True, bottom=False)
    g.tick_params(direction="out", length=2, width=1, colors="k", which="minor", left=True, bottom=False)
    ax.axhline(0, linestyle='-', color="k")
    ax.set_xlabel(None)
    ax.get_xaxis().set_ticks([])
    ax.set_ylim([-.2, .2])
    ax.set_title(title)
    if not label:
        ax.set_ylabel(None)
        ax.get_yaxis().set_ticks([])

    return g

def viz_all_variance(loc, growth_upper_bound, growth_lower_bound, dist_upper_bound, dist_lower_bound, posterior, newcomer_subreddits, newcomer_sub_diffs, newcomer_dists):
    sns.set_style(style="white")
    fig, axes = plt.subplots(1, 2, figsize = (15, 5))
    palette = ["#d3d3d3", "#fee143", "#4ac6ff", "#ff4a7d"]
    mean_palette = ["#eeeeee", "#ffeb80", "#05b1ff", "#ff7ea2"]
    sns.despine(left=False, bottom=True)
    g_high = np.nonzero(newcomer_dists < dist_upper_bound)[0]
    g_low = np.nonzero(newcomer_dists > dist_lower_bound)[0]

    g_high_sample  = np.random.choice(g_high, replace=False, size=(40,))
    g_low_sample  = np.random.choice(g_low, replace=False, size=(40,))

    viz_variance(axes[0], g_low, g_low_sample, posterior, newcomer_subreddits, newcomer_sub_diffs, newcomer_dists, palette[2], mean_palette[2], label=True, title = "Low Distinctiveness")
    viz_variance(axes[1], g_high, g_high_sample, posterior, newcomer_subreddits, newcomer_sub_diffs, newcomer_dists, palette[3], mean_palette[3], label=False, title = "High Distinctiveness")

    plt.savefig(loc)

def viz(loc, high_percentile, low_percentile, inf_data, newcomer_subreddits, newcomer_sub_diffs, newcomer_dists):
    #reformat posterior a bit
    posterior = inf_data["posterior"]
    posterior_dict = {}
    posterior_dict["sub_pop_effects"] = np.array(posterior["sub_pop_effects"])
    shape = posterior_dict["sub_pop_effects"].shape
    posterior_dict["sub_pop_effects"] = np.reshape(posterior_dict["sub_pop_effects"], (shape[0]*shape[1], shape[2], shape[3]))

    posterior_dict["growth_effect"] = np.array(posterior["growth_effect"])
    shape = posterior_dict["growth_effect"].shape
    posterior_dict["growth_effect"] = np.reshape(posterior_dict["growth_effect"], (shape[0]*shape[1], shape[2], shape[3]))

    posterior_dict["phis"] = np.array(posterior["phis"])
    shape = posterior_dict["phis"].shape
    posterior_dict["phis"] = np.reshape(posterior_dict["phis"], (shape[0]*shape[1], shape[2]))

    growth_upper_bound, growth_lower_bound = np.percentile(newcomer_sub_diffs, low_percentile), np.percentile(newcomer_sub_diffs, high_percentile)
    dist_upper_bound, dist_lower_bound = np.percentile(newcomer_dists, low_percentile), np.percentile(newcomer_dists, high_percentile)
    viz_all_variance(loc, growth_upper_bound, growth_lower_bound, dist_upper_bound, dist_lower_bound, posterior, newcomer_subreddits, newcomer_sub_diffs, newcomer_dists)
