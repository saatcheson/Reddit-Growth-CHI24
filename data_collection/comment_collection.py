"""
The purpose of this script is to get all comments made by 2018-2019 subreddits and a
random sampling of comments from other subreddits.
"""


import os
import json
import random
import subprocess


# create sub-directories in data directory
data_dir = '...' # path to base data directory
folders = ['sample_comments', 'finetuning_comments', 'authors', 'avg_subreddit']
for folder in folders:
    cmd = f'mkdir {data_dir}/{folder}'
    os.system(cmd)

## settings for ingestion of fine-tuning comments ##
odds = 50  # for 1:50 odds that a comment is selected for fine-tuning
training_cap = 80_000_000  # max number of comments for distinctiveness fine-tuning

# STAGE 0: read in names of 2018-2019 sample subreddits

infile_2018 = '2018_subscriber_stats.csv'
infile_2019 = '2019_subscriber_stats.csv'

with open(infile_2018, 'r') as f1, open(infile_2019, 'r') as f2:
    subreddits_2018 = set([line.strip().split(',')[0] for line in f1.readlines()])
    subreddits_2019 = set([line.strip().split(',')[0] for line in f2.readlines()])
    subreddits = subreddits_2018.union(subreddits_2019)

    for year in range(2018, 2022):

        if year == 2018:
            init_month = 3
        else:
            init_month = 1

        for month in range(init_month, 13):

            data = {'sample_comments': [],
                    'finetuning_comments': [],
                    'authors': []}

            # STAGE 1: retrieve, unzip dump file

            cmd = f'python3 download-unzip_dumpfile.py {year} {month} comments'
            os.system('nohup ' + cmd + ' &>/dev/null &')

            # STAGE 2: parse comments

            dumpfile_dir = '...' # base directory where unzipped dump files are located
            if 1 <= month <= 9:
                dumpfile = f'{dumpfile_dir}/RC_{year}-0{month}'
            else:
                dumpfile = f'{dumpfile_dir}/RC_{year}-{month}'

            with open(dumpfile, 'r') as dump:
                for datum in dump:
                    datum = json.loads(datum.strip())

                    if datum['body'] == '[removed]':
                        status = 'removed'
                    elif datum['body'] == '[deleted]':
                        status = 'deleted'
                    else:
                        status = 'alive'

                    document = {'subreddit': datum['subreddit'],
                                'author': datum['author'],
                                'id': datum['id'],
                                'created_utc': int(datum['created_utc']),
                                'status': status,
                                'body': datum['body']}

                    author = {'subreddit': datum['subreddit'],
                                   'author': datum['author']}

                    # STAGE 3: save comments

                    data['authors'].append(author)
                    if datum['subreddit'] in subreddits:
                        data['sample_comments'].append(document)
                    else:
                        if len(data['finetuning_comments']) < training_cap:
                            if random.randint(1, odds) == 1:
                                data['finetuning_comments'].append(document)

            # randomly sample 100,000 comments to consider for "average" subreddit representation

            cmd = ['wc', '-l', dumpfile]
            ncomments = int(subprocess.run(cmd, stdout=subprocess.PIPE, text=True).stdout.strip().split(' ')[0])
            avg_subreddit = f'{data_dir}/avg_subreddit/{year}_{month}'

            nsamples = 100_000
            cmd = "nohup cat " + dumpfile + \
                  " | awk 'BEGIN {srand()} !/^$/ { if (rand() <= " + str(nsamples / ncomments) + \
                  ") print $0}' > " + f'{avg_subreddit} &'
            os.system(cmd)

            # STAGE 3: remove unzipped dump file

            cmd = f'rm {dumpfile}'
            os.system('nohup ' + cmd + ' &>/dev/null &')

            # STAGE 4: save data to sub-directories

            for d in data:
                with open(f'{data_dir}/{d}/{year}_{month}.json', 'w') as f:
                    json.dump(data[d], f)