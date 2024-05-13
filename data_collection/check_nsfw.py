'''
[check_nsfw.py]
The purpose of this script is to check the NSFW status of subreddits. This script uses the Python Reddit
API Wrapper (PRAW) to query Reddit for checking the status of a subreddit. You will need to configure
the code with your own account information. This script takes as input a file of unique subreddit names,
where each line is the name of a subreddit. This scripts outputs a JSON file containing info about the
NSFW status of all input subreddits ("1" for NSFW and "0" for SFW).
'''


import json
from Reddit import Reddit


try:
    reddit = Reddit().connect_to_reddit()
except Exception as e:
    print(e)
    exit(1)

subreddits = '...' # path to input file containing subreddit names
nsfw_status = '.../nsfw.json' # path to output file containing NSFW statuses
with open(subreddits, 'r') as f1, open(nsfw_status, 'w') as f2:
    nsfw_info = {}
    subreddits = set([l.strip() for l in f1.readlines()])
    for s in subreddits:
        try:
            if reddit.subreddit(s).over18:
                nsfw_info[s] = 1
            else:
                nsfw_info[s] = 0
        except Exception as e:
            print(e)
            nsfw_info[s] = -1
    json.dump(nsfw_info, f2)