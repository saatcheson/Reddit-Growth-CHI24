"""
The purpose of this script is for tracking end-of-month subscriber count for communities.

=======================
How to use in a script:
=======================
from subscribers import Subscribers

> subscribers = Subscribers()                             # instantiate Subscribers object, load latest data from subscribers_master.json
> data = load_dump_submissions()                          # read submissions from dump file - data should be a list of dictionary objects, where each dictionary represents a single submission
> subscribers.update(data)                                # will update the subscriber info based on the new data
> subscribers.save_master()                               # save latest version of subscribers data to subscribers_master.json

*** NOTE ***
- latest data persists in subscribers.json, so code is configured for use across multiple runs.
Just make sure to load the subscriber.json file at the beginning of the script and save at the end.
"""

"""
Format of the subscribers.json file

{
    subreddit_name: {
        start: {
            year: 2018,
            month: 5
        },
        end: {
            year: 2020,
            month: 5
        },
        2018: {
            5: {
                subscribers: 5000,
                created_utc: 13049203
            },
            ...
        },
        2019: {
            1: {
                subscribers: 25693,
                created_utc: 13042324
            },
            ...
        },
        2020: {
            ...,
            5: {
                subscribers: 109572,
                created_utc: 13041048
            }
        }
    },
    ...
}
"""


class Subscribers:
    def __init__(self):
        import os
        self.master = None
        if not os.path.exists('subscribers.json'):
            os.system('touch subscribers.json')
        self.load_master()

    def _dates_to_scrape(self, created_utc):
        import datetime as dt
        year = dt.datetime.utcfromtimestamp(created_utc).year
        month = dt.datetime.utcfromtimestamp(created_utc).month
        end_month = month
        end_year = year + 2
        dates = []
        done = False
        for i in range(year, end_year + 1):
            for j in range(month, 13):
                dates.append((i, j))
                if j == end_month and i == end_year:
                    done = True
                    break
                if j == 12:
                    month = 1
            if done:
                break
        return dates

    def load_master(self):
        import json
        with open('subscribers.json', 'r') as f:
            self.master = json.load(f)

    def save_master(self):
        import json
        with open('subscribers.json', 'r') as f:
            json.dump(self.master, f)

    def update(self, submissions):  # 'submissions' should be a list submissions, where each comment is a Python dictionary
        import datetime as dt
        for s in submissions:
            subreddit = s['subreddit']
            created_utc = int(s['created_utc'])

            month = dt.datetime.utcfromtimestamp(created_utc).month
            year = dt.datetime.utcfromtimestamp(created_utc).year

            if subreddit not in self.master:
                self.master[subreddit] = {}
                dates = self._dates_to_scrape(created_utc)
                self.master[subreddit]['start'] = {'year': dates[0][0], 'month': dates[0][1]}
                self.master[subreddit]['end'] = {'year': dates[-1][0], 'month': dates[-1][1]}
                for date in dates:
                    if date[0] not in self.master[subreddit]:
                        self.master[subreddit][date[0]] = {}
                    self.master[subreddit][date[0]][date[1]] = {}
                    self.master[subreddit][date[0]][date[1]]['subscribers'] = -1
                    self.master[subreddit][date[0]][date[1]]['created_utc'] = -1

            if self.master[subreddit][year][month]['created_utc'] < created_utc:
                self.master[subreddit][year][month]['subscribers'] = s['subreddit_subscribers']
                self.master[subreddit][year][month]['created_utc'] = created_utc