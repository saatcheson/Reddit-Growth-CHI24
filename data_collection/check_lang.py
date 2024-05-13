"""
[check_lang.py]
The purpose of this script is for detecting English comments in a subreddit. The input to the script
is a JSON file containing a list of comments represented as JSON objects (refer to comment_collection.py to see comment
object details). The output of the script is a JSON file, containing information about (1) the number
of total comments in a subreddit and (2) the number of comments where English was the detected language in the subreddit.
"""


import json
from langdetect import detect


comments_file = '...' # path to JSON file containing comments to be analyzed
lang_file = '.../lang.json' # desired path to output JSON file containing language information about each subreddit from "comments_file"

with open(comments_file, 'r') as f1, open(lang_file, 'w') as f2:
    lang_info = {}
    comments = json.load(f1)
    for c in comments:
        if c['subreddit'] not in lang_info:
            lang_info[c['subreddit']] = {'total': 0, 'en': 0}
        lang_info[c['subreddit']]['total'] += 1
        try:
            detected_lang = detect(c['body'])
            if detected_lang == 'en':
                lang_info[c['subreddit']]['en'] += 1
        except:
            pass
    json.dump(lang_info, f2)