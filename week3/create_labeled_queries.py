import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

min_queries = 1

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=min_queries,  help=f"The minimum number of queries per category label (default is {min_queries})")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME:
def normalize(query):
  # Convert all letters to lowercase.
  low = query.lower()
  # Treat any character that is not a number or letter as a space.
  spaces = re.sub('[\W_]', ' ', low)
  words = spaces.split(' ')
  # Trim multiple spaces (which may result from previous step) to a single space.
  noSpace = [word for word in words if len(word) > 0]
  # Use the nltk stemmer to stem all query tokens.
  stems = [stemmer.stem(word) for word in noSpace]
  return ' '.join(stems)

queries_df['query'] = queries_df['query'].apply(normalize)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
#
# Implementation: roll up only the leaves, then prune the leaves, and repeat
# This is to avoid rolling up non-leaves that won't need to be rolled up once their leaves are rolled up
#
all_counts = queries_df.groupby('category').size().reset_index(name='rollup_count')
all_counts['rolledup_categories'] = all_counts[['category']].values.tolist()
enough_counts = pd.DataFrame()
full = pd.merge(parents_df, all_counts)

def combineLists(l1, l2, c2):
  if c2 > 0: return l1.extend(l2)
  else: return l1

while True:
  # find leaves among rollups
  leaves = full[~full['category'].isin(full['parent'].values)]
  trunc  = full[full['category'].isin(full['parent'].values)]

  leaves_enough_counts = leaves[leaves['rollup_count'] >= min_queries]
  if len(leaves_enough_counts) > 0:
    enough_counts = pd.concat([enough_counts, leaves_enough_counts])
  leaves_to_rollup = leaves[leaves['rollup_count'] < min_queries]
  # print(f"Enough counts size: {len(enough_counts)}   Trunc size: {len(trunc)}    Leaves to rollup: {len(leaves_to_rollup)}")

  if len(trunc) > 0 and len(leaves_to_rollup) > 0:
    full = pd.merge(trunc, leaves_to_rollup, left_on='category', right_on='parent', how='left').drop(['category_y','parent_y'], axis=1) \
    .rename(columns={'category_x': 'category', 'parent_x': 'parent'})
    full['rollup_count_y'] = full['rollup_count_y'].fillna(0)
    full['rollup_count'] = full['rollup_count_x'] + full['rollup_count_y']
    full.apply(lambda x: combineLists(x.rolledup_categories_x, x.rolledup_categories_y, x.rollup_count_y), axis=1)
    full.drop(['rollup_count_x','rollup_count_y','rolledup_categories_y'], axis=1, inplace=True)
    full = full.rename(columns={'rolledup_categories_x': 'rolledup_categories'})
  elif len(leaves_to_rollup) > 0:
    enough_counts = pd.concat([enough_counts, leaves_to_rollup]) # if not enough counts in whole tree
    break
  else:
    break

print("out of while")

# Explode hangs, so do it in batches
os.remove(output_file_name)
batchSz = 2
count = -(len(enough_counts) // -batchSz)
for i in range(count):
  # print("before explode")
  enough_counts2 = enough_counts.iloc[i*batchSz:(i+1)*batchSz].explode("rolledup_categories")
  # print("after explode")
  queries_df2 = pd.merge(queries_df, enough_counts2, left_on='category', right_on='rolledup_categories')
  queries_df2 = queries_df2[['category_y', 'query']].rename(columns={'category_y': 'category'})

  # Create labels in fastText format.
  queries_df2['label'] = '__label__' + queries_df2['category']

  # Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
  queries_df2 = queries_df2[queries_df2['category'].isin(categories)]
  queries_df2['output'] = queries_df2['label'] + ' ' + queries_df2['query']
  print(f"Output count: {len(queries_df2)}")
  queries_df2[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', mode='a', quoting=csv.QUOTE_NONE, index=False)
  print(f"Wrote batch {i}\n")
