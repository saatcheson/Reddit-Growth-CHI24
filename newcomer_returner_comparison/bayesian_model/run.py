from load import read_all, gen_or_load_df, extract_outcomes, extract_and_std_predictors, gen_masks
from model import run_model
from viz import viz


#read in info from shared data folder
skip_list, og_dist_scores, subscriber_counts = read_all("../../data/skip_list.txt", "../../data/distinctiveness_scores.json", "../../data/subscriber_info.csv")

#read if exists (or generate and write) df of newcomer/return scores
new_ret_df = gen_or_load_df("../outputs/scores.csv", subscriber_counts, og_dist_scores, skip_list)
num_subs = max(new_ret_df.subreddit_id) + 1

#outcome variables
output_var_dict = extract_outcomes(new_ret_df)

#predictor variables
predictor_var_dict = extract_and_std_predictors(new_ret_df)

#masks
mask_dict = gen_masks(predictor_var_dict)

#run the model
inf_data = run_model(output_var_dict, predictor_var_dict, mask_dict, num_subs)

#generates Figure 5 in the paper
viz("../outputs/newcomer_gap_variation.pdf", 75, 25, inf_data, predictor_var_dict["newcomer_subreddits"], predictor_var_dict["newcomer_sub_diffs"], predictor_var_dict["newcomer_dists"])
