import praw
import json


class Reddit:
    def __init__(self, config_file="praw_config.json"):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.config = config

    def connect_to_reddit(self):
        return praw.Reddit(
            client_id = self.config["CLIENT_ID"],
            client_secret = self.config["CLIENT_SECRET"],
            password= self.config["REDDIT_PW"],
            user_agent = self.config["USER_AGENT"],
            username = self.config["REDDIT_USER"]
        )