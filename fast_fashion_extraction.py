#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install --upgrade praw


# In[ ]:


import praw
import csv
import time

reddit = praw.Reddit(client_id='DR_ig9gctxjs727ZiKLchg',
                     client_secret='dw2BtmjFJqTd0voNJmxGSswZGX0g_g',
                     user_agent="Antiwork Scraper 1.0 by /u/bbdadson11")

subreddit_name = 'Anticonsumption'
posts_limit = 1000  # Number of posts to fetch per request
total_posts = 20000  # Total number of posts to scrape
search_term = 'fast fashion'  # The word to search for in the post titles

subreddit = reddit.subreddit(subreddit_name)

posts = []
num_posts = 0
before = None

while num_posts < total_posts:
    # Fetch posts
    new_posts = list(subreddit.search(search_term, limit=posts_limit, params={'before': before}))
    num_new_posts = len(new_posts)

    # Break the loop if there are no more posts
    if num_new_posts == 0:
        break

    # Determine the last post's ID to use as the 'before' parameter in the next request
    before = new_posts[-1].id

    # Add the new posts to the list
    posts.extend(new_posts)

    num_posts += num_new_posts

    # Add a delay between requests
    time.sleep(2)  # Adjust the delay as needed

# Extract relevant information from each post
extracted_posts = []
for post in posts:
    author_name = post.author.name if post.author else 'Unknown'  # Handle NoneType error
    extracted_post = {
        'title': post.title,
        'author': author_name,
        'created_utc': post.created_utc,
        'selftext': post.selftext,
        'score': post.score,
        'upvote_ratio': post.upvote_ratio,
        'num_comments': post.num_comments,
        'url': post.url
        # Add more fields as needed
    }
    extracted_posts.append(extracted_post)

# Save extracted posts to a CSV file
csv_file = 'final_fashion.csv'
fieldnames = ['title', 'author', 'created_utc', 'selftext', 'score', 'upvote_ratio', 'num_comments', 'url']  # Add more field names as needed

with open(csv_file, mode='w', encoding='utf-8-sig', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(extracted_posts)

print(f'Successfully extracted {len(extracted_posts)} posts with the word "fast fashion" and saved to {csv_file}.')


# In[ ]:




