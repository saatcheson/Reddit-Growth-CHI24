import json
import numpy as np

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
def _read_subscribers(subscribers_loc, skip_list, valid_subs):
    subscriber_data = {}
    with open(subscribers_loc) as f:
        for line in f:
            entry = line.strip().split(",")
            sub = entry[0]
            if sub in valid_subs and sub not in skip_list:
                raw =  [int(num) for num in entry[2:]]
                new_list = []
                for num in raw:
                    if num >= 0:
                        new_list.append(np.log(num))
                    else:
                        new_list.append(num)
                subscriber_data[sub] = new_list
    return subscriber_data

def _read_active_users(active_user_loc, skip_list, valid_subs):
    active_users = {}
    with open(active_user_loc) as f:
        for line in f:
            entry = line.split(",")
            sub = entry[0]
            counts = [int(elem) for elem in entry[2:]]
            if sub in valid_subs and sub not in skip_list:
                active_users[sub] = counts
    return active_users

def _read_diversity(diversity_loc, skip_list, valid_subs):
    with open(diversity_loc) as f:
        diversity_data = json.load(f)
    to_remove = []
    for key in diversity_data.keys():
        if key in skip_list or key not in valid_subs:
            to_remove.append(key)
    for key in to_remove:
        diversity_data.pop(key)
    return diversity_data

def _read_months(month_loc, skip_list, valid_subs):
    with open(month_loc) as f:
        month_indices = json.load(f)
    to_remove = []
    for key in month_indices.keys():
        if key in skip_list or key not in valid_subs:
            to_remove.append(key)
    for key in to_remove:
        month_indices.pop(key)
    return month_indices

def _read_removal_ratio(removal_loc, skip_list, valid_subs):
    removal_data = {}
    with open(removal_loc) as removal_rate_file:
        for line in removal_rate_file:
            entry = line.strip().split(",")
            sub = entry[0]
            if sub in valid_subs and sub not in skip_list:
                removal_rates = [float(val) for val in entry[2:]]
                removal_data[sub] = removal_rates
    return removal_data
def read_all(skip_loc, dist_scores_loc, subscribers_loc,
        active_users_loc, diversity_loc, removal_loc, month_loc):
    skip_list = _read_skip(skip_loc)
    distinctiveness_data = _read_distinctiveness(dist_scores_loc, skip_list)
    valid_subs = set(distinctiveness_data.keys())
    subscriber_data = _read_subscribers(subscribers_loc, skip_list, valid_subs)
    active_users = _read_active_users(active_users_loc, skip_list, valid_subs)
    diversity_data =  _read_diversity(diversity_loc, skip_list, valid_subs) 
    removal_ratio = _read_removal_ratio(removal_loc, skip_list, valid_subs) 
    month_indices = _read_months(month_loc, skip_list, valid_subs)
    return subscriber_data, distinctiveness_data, month_indices, active_users, diversity_data, removal_ratio

def _longest_contig(subscriber_timeline, distinctiveness_timeline, diversity_timeline):
    num_total = len(subscriber_timeline)
    longest = 1
    cur_longest = 1
    longest_start = 0
    longest_end = 1
    cur_start = 0
    for i, val in enumerate(subscriber_timeline):
      a = (val != -1) and (val != -1.0)
      b = (distinctiveness_timeline[i] != -1) and (distinctiveness_timeline[i] != -1.0)
      c = (diversity_timeline != -1) and (diversity_timeline[i] != -1.0)
      if a and b and c:
        if cur_start == -1:
            cur_start = i
        cur_longest += 1
      else:
        if cur_longest > longest:
          longest_start = cur_start
          longest_end = i -1
          longest = cur_longest
          cur_longest = 1
        cur_start = -1
    if cur_longest > longest:
        longest_start = cur_start
        longest_end = i+1
        longest = cur_longest
    return longest_start, longest_end

def _trim(timeline, start, end):
    num_total = len(timeline)
    return np.concatenate([timeline[start:end], np.full(num_total-(end-start), -1)], axis=0)
    
def convert_to_mat(subscriber_data, distinctiveness_data, diversity_data, removal_data, month_data):
    subs = []
    subscriber_rows = []
    distinctiveness_rows = []
    removal_rows = []
    diversity_rows = []
    month_rows = []
    for ind, sub in enumerate(subscriber_data.keys()):
        start, end = _longest_contig(subscriber_data[sub], distinctiveness_data[sub], diversity_data[sub])
        if end - start > 1:
            subs.append(sub)
            subscriber_rows.append(_trim(subscriber_data[sub], start, end))
            distinctiveness_rows.append(_trim(distinctiveness_data[sub], start, end))
            removal_rows.append(_trim(removal_data[sub], start, end))
            diversity_rows.append(_trim(diversity_data[sub], start, end))
            month_rows.append(_trim(month_data[sub], start, end))
    s_mat = np.stack(subscriber_rows)
    d_mat = np.stack(distinctiveness_rows)
    r_mat = np.stack(removal_rows)
    div_mat = np.stack(diversity_rows)
    month_mat = np.stack(month_rows)    
    return subs, s_mat, d_mat, r_mat, div_mat, month_mat

def gen_mask(s_mat, d_mat, r_mat, div_mat):
    return (s_mat != -1) * (d_mat != -1) * (r_mat != -1) * (div_mat != -1)

def _diff_transform(mat):
    subtract_mat = np.concatenate([np.zeros((mat.shape[0], 1)), mat[:, 0:-1]], axis=1)
    differences_mat = np.where(mat != -1, mat - subtract_mat, mat)
    return differences_mat
    
def standardize_valid(mat):
    valid_mean = np.mean(mat[mat != -1])
    valid_std = np.std(mat[mat != -1])
    return np.where(mat != -1, (mat - valid_mean)/valid_std, mat), valid_mean, valid_std

def standardize_front_mat(mat):
    zero_start = np.zeros((mat.shape[0], 1))
    standardized_mat, mean, std = standardize_valid(mat[:, 1:])
    diff_mat = np.concatenate([zero_start, standardized_mat], axis=1)
    return diff_mat, mean, std
    
def standardize_and_diff_mat(mat, mean_dict, std_dict, name, diff=True):
    if diff:
        mat = _diff_transform(mat)
    standardized_mat, mean, std = standardize_front_mat(mat)
    mean_dict[name], std_dict[name] = mean, std
    return standardized_mat, mean_dict, std_dict
    



    

    
    
    
    
    
    







