"""
[check_bot.py]
This script is for marking author names as "1" if they are suspected bot and "0" otherwise.
The input to this script should be a list of author names (names need not be unique). The
output of the script will be a file containing all unique author names and a "1" if they
are a suspected bot and "0" otherwise. Each row of the file is then "author_name,1" or
"author_name,0".
"""

author_names = '...' # desired path to file with author names in CSV format (e.g., authorA,authorB,authorC,...)
author_suspected = '.../suspected.csv' # desired path to CSV file with author names and suspected bot statuses (e.g., authorA,1)

with open(author_names, 'r') as f1, open('bot_substring.txt', 'r') as f2, open(author_suspected, 'w') as f3:

    words = [line.strip().lower() for line in f2.readlines()]
    authors = [line.strip().lower() for line in f1.readlines()]

    bots = set()
    not_bots = set()

    for author in authors:
        c1 = 'bot' in author.lower()
        c2 = not 'boT' in author
        c3 = len(author) >= 3
        c4 = True
        for word in words:
            if word in author.lower():
                c4 = False
                break
        c5 = author.lower() in ['b0trank', '[deleted]', 'savevideo', 'automoderator']

        if c3:
            if (c1 and c2 and c4) or c5:
                bots.add(author)
            else:
                not_bots.add(author)
        else:
            not_bots.add(author)

    for author in bots:
        f3.write(f'{author},1\n')
    for author in not_bots:
        f3.write(f'{author},0\n')