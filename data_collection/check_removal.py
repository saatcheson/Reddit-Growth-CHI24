"""
[check_removal.py]
The purpose of this script is to verify the removal status of Pushshift comments with
their current status through PRAW. The input to this script is a file containing
comment IDs, with each row of the file corresponding to a single comment ID.
The output of this file is a JSON file containing info about the current status of the comment
(i.e., alive, removed, deleted)
"""


import json
import time
import math
from Reddit import Reddit


comment_ids = '...' # path to input file containing comment IDs
removal_status = '.../removal_status.json' # path to output file containing removal status associated with input IDs

try:
    reddit = Reddit().connect_to_reddit()
except Exception as e:
    print(e)
    exit(1)

status_info = {}

with open(comment_ids, 'r') as f1, open(removal_status, 'w') as f2:
    ids = [l.strip() for l in f1.readlines()]
    chunks = math.ceil(len(ids) / 100)
    for i in range(chunks):
        success = False
        while not success:
            try:
                for c in reddit.info(fullnames=ids[i * 100 : (i+1) * 100]):
                    status = 0  # alive
                    if c.body == '[removed]' or c.author == '[removed]':
                        status = 1  # removed
                    elif c.body == '[deleted]' or c.author == '[deleted]':
                        status = 2  # deleted
                    if c.subreddit not in status_info:
                        status_info[c.subreddit] = []
                    status_info[c.subreddit].append({'id': c.id, 'status': status})
                success = True
            except Exception as e:
                print(e)
                time.sleep(1)
    json.dump(status_info, f2)