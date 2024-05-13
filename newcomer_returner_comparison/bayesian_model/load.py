import json
import numpy as np
import os
import numpy as np
import pandas as pd
def _read_skip(skip_loc):
    SKIP_LIST = set()
    with open(skip_loc, 'r') as f:
        for line in f:
            SKIP_LIST.add(line.strip())
    return SKIP_LIST

def _read_distinctiveness(dist_scores_loc, skip_list):
    with open(dist_scores_loc) as f:
        distinctiveness_data = json.load(f)
    for sub in skip_list:
        if sub in distinctiveness_data:
            distinctiveness_data.pop(sub)
    return distinctiveness_data

def _read_subscribers(subscribers_loc, skip_list):
    subscriber_data = {}
    with open(subscribers_loc) as f:
        for line in f:
            entry = line.split(",")
            sub = entry[0]
            if sub not in skip_list:
                entry = [int(val) for val in entry[2:]]
                diffs = [0] + [np.log(entry[i] + 1) - np.log(entry[i-1]+1) if (entry[i] != -1 and entry[i-1] != -1) else -1 for i in range(1, len(entry))]
                subscriber_data[sub] = diffs
    return _standardize_subscribers(subscriber_data)


def _standardize_subscribers(subscriber_data):
    vals = []
    for key in subscriber_data.keys():
        row = subscriber_data[key]
        for val in row:
            if val != -1:
                vals.append(val)
    mean = np.mean(vals)
    std = np.std(vals)
    for key in subscriber_data.keys():
        subscriber_data[key] = [(val-mean)/std if val != -1 else val for val in subscriber_data[key] ]
    return subscriber_data

#read and return the skip list, original (i.e. not newcomer/return split) distinctiveness scores, and the subscriber count data
def read_all(skip_loc, dist_loc, sub_loc):
    skip_list = _read_skip(skip_loc)
    dist = _read_distinctiveness(dist_loc, skip_list)
    subs = _read_subscribers(sub_loc, skip_list)
    return skip_list, dist, subs

# count how many newcomer/non-newcomer comments are available for a specific sub, at a specific time step
# Returns 0 if the file is empty, and -1 if there is no data at that timestep
def _num_comments_at_timestep(subreddit, date, newcomer=True):
    if newcomer:
        path = "../data/newcomer_comments/{}/{}.json".format(subreddit, date)
    else:
        path = "../data/returner_comments/{}/{}.json".format(subreddit, date)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return len(json.load(f)["authors"])
        except json.JSONDecodeError as e:
            return 0
    else:
        return -1
    
# maps data from  "sample-newcomer-comments" and "sample-non-newcomer-comments" files to a record used to create pandas dataframe
# Each .json contains two "score" entries per subreddit -- a "super" score comparing the newcomer/non-newcomer centroid to the reddit-wide
# centroid from that month, and a "subreddit" score, comparing the newcomer/non-newcomer centroid to the subreddit's centroid from that month
# returns 0 if no data, and 1 on success
def _gen_entries(records, sub_data, subscriber_data, distinctiveness, subreddit, user2id, sub2id, comment_count, relative_time_step, date, absolute_time_step, newcomer=True):
    num_subs = subscriber_data[subreddit]
    dist_score = distinctiveness[subreddit][relative_time_step]
    if dist_score != -1:
        dist_score = 1- dist_score
    if sub_data[date] == {}:
        return 0
    records.append({
        "subreddit_id": sub2id[subreddit],
        "absolute_time_step": absolute_time_step,
        "relative_time_step": relative_time_step,
        "subscriber_diff": subscriber_data[subreddit][relative_time_step],
        "base_dist": dist_score,
        "newcomer_status": newcomer,
        "super": False,
        "score": sub_data[date][0]
    })
    records.append({
        "subreddit_id": sub2id[subreddit],
        "absolute_time_step": absolute_time_step,
        "relative_time_step": relative_time_step,
        "subscriber_diff": subscriber_data[subreddit][relative_time_step],
        "base_dist": dist_score,
        "newcomer_status": newcomer,
        "super": True,
        "score": sub_data[date][1]
    })
    return records

# iterate over newcomer/non-newcomer distinctiveness files and write info into a pandas dataframe    
def gen_df(subscriber_data, distinctiveness, skip_list):
    """
    Data frame structure

    comment_id | user_id | subreddit_id | absolute_time_step | author_num_comment | subreddit_size | newcomer_status | super | score
    int        | int     | int          | int (0-46)         | int (0-10)         | int            | Bool            | Bool  | float (0-1)

    """
    comment_count = 0
    absolute_time_step = 0
    user2id = {}
    sub2id = {}
    id2sub = {}
    records = []
    for subreddit in subscriber_data.keys():
        if subreddit not in skip_list:
            paths = {
                "newcomer": "../data/newcomer_scores-v4/{}.json".format(subreddit),
                "returner": "../data/returner_scores-v4/{}.json".format(subreddit)
            }
            for score_type in ["newcomer", "returner"]:
                months_seen = -1
                absolute_time_step = 0
                cur_path = paths[score_type]
                if os.path.exists(cur_path):
                    with open(cur_path) as f:
                        sub_data = json.load(f)
                    for year in range(2018, 2022):
                        start = 1
                        if year == 2018:
                            start = 3 
                        for month in range(start, 13):
                            date = "{}-{}".format(year, month)
                            newcomer_comments = _num_comments_at_timestep(subreddit, date, newcomer=True)
                            returner_comments = _num_comments_at_timestep(subreddit, date, newcomer=False)
                            if newcomer_comments != -1:
                                # Third condition skips the first month of data for every subreddit, since no subreddits will have returning users that month
                                if newcomer_comments >= 50 and returner_comments >= 50 and months_seen >= 0:
                                    if subreddit not in sub2id:
                                        sub2id[subreddit] = len(sub2id)
                                        id2sub[sub2id[subreddit]] = subreddit
                                    cur_sub_id = sub2id[subreddit]
                                    records = _gen_entries(records, sub_data, subscriber_data, distinctiveness, subreddit, user2id,sub2id, comment_count, months_seen, date, absolute_time_step, newcomer=score_type=="newcomer")
                                    if score_type == "newcomer":
                                        comment_count += newcomer_comments
                                    else:
                                        comment_count += returner_comments
                                months_seen += 1
                            absolute_time_step += 1
    absolute_time_step += 1
    return pd.DataFrame.from_records(records)

# loads the dataframe containing newcomer/returner scores from loc if it has already been written to file
# else, generates a new one and writes it to loc
def gen_or_load_df(loc, subscribers, distinctiveness, skip_list):
    if os.path.exists(loc):
        new_ret_df = pd.read_csv(loc)
    else:
        new_ret_df = gen_df(subscribers, distinctiveness, skip_list)
        with open(loc, "w+") as f:
            new_ret_df.to_csv(f)
    return new_ret_df
     
    
def extract_outcomes(new_ret_df):
    newcomer_select = (new_ret_df.super == True) & (new_ret_df.newcomer_status == True)
    returner_select = (new_ret_df.super == True) & (new_ret_df.newcomer_status == False)
    newcomer_super_scores = np.array(new_ret_df[newcomer_select].score)
    returner_super_scores = np.array(new_ret_df[returner_select].score)
    return {"newcomer_super_scores": newcomer_super_scores, "returner_super_scores": returner_super_scores}

def _standardize(values):
    valid = [val for val in values if val != -1]
    values_mean = np.mean(valid)
    values_std = np.std(valid)
    values = np.array([(val - values_mean)/values_std if val != -1 else val for val in values])
    return values 

#not all predictors extracted by this function actually get used in model
def extract_and_std_predictors(new_ret_df):
    newcomer_select = (new_ret_df.super == True) & (new_ret_df.newcomer_status == True)
    returner_select = (new_ret_df.super == True) & (new_ret_df.newcomer_status == False)
    #categorical newcomer preds
    newcomer_abs_timesteps = np.array(new_ret_df[newcomer_select].absolute_time_step)
    newcomer_rel_timesteps = np.array(new_ret_df[newcomer_select].relative_time_step)
    newcomer_subreddits = np.array(new_ret_df[newcomer_select].subreddit_id)
    #categorical returner preds
    returner_abs_timesteps =  np.array(new_ret_df[returner_select].absolute_time_step)
    returner_rel_timesteps =  np.array(new_ret_df[returner_select].relative_time_step)
    returner_subreddits = np.array(new_ret_df[returner_select].subreddit_id)

    #newcomer/returner growth numbers
    newcomer_sub_diffs = np.array(new_ret_df[(new_ret_df.super == True) & (new_ret_df.newcomer_status == True)].subscriber_diff)
    returner_sub_diffs = np.array(new_ret_df[returner_select].subscriber_diff)
    
    #newcomer/returner distinctiveness scores
    newcomer_dists = np.array(new_ret_df[(new_ret_df.super == True) & (new_ret_df.newcomer_status == True)].base_dist)
    returner_dists = np.array(new_ret_df[returner_select].base_dist)

    #standardize growth numbers
    newcomer_sub_diffs = _standardize(newcomer_sub_diffs)
    returner_sub_diffs = _standardize(returner_sub_diffs)
      
    #standardize distinctiveness scores
    newcomer_dists = _standardize(newcomer_dists)
    returner_dists = _standardize(returner_dists)
    
    return {
        "newcomer_abs_timesteps": newcomer_abs_timesteps,
        "newcomer_rel_timesteps": newcomer_rel_timesteps,
        "newcomer_subreddits": newcomer_subreddits,
        "newcomer_sub_diffs": newcomer_sub_diffs,
        "returner_abs_timesteps": returner_abs_timesteps,
        "returner_rel_timesteps": returner_rel_timesteps,
        "returner_subreddits": returner_subreddits,
        "returner_sub_diffs": returner_sub_diffs,
        "newcomer_dists": newcomer_dists,
        "returner_dists": returner_dists
    }

#generates masks used to ignore months where there's insufficient/no data
def gen_masks(predictor_var_dict):
    newcomer_mask = (predictor_var_dict["newcomer_dists"] != -1)*(predictor_var_dict["newcomer_sub_diffs"] != -1)
    returner_mask = (predictor_var_dict["returner_dists"] != -1)*(predictor_var_dict["returner_sub_diffs"] != -1)
    return {"newcomer_mask": newcomer_mask, "returner_mask": returner_mask}
    
    
    
    
