import pandas as pd
import re
import numpy as np
import csv

minCount = 10 # 500

inP = "/workspace/datasets/fasttext/labeled_products.txt"
outP = "/workspace/datasets/fasttext/pruned_labeled_products.txt"

with open(inP) as inF:
    lines = list(inF)
    tupArr = []
    for line in lines[:100]:
        catNname = re.search('([^ ]+) +(.+)', line)
        cat = catNname.group(1)
        name = catNname.group(2)
        tupArr.append((cat,name.rstrip()))
    df = pd.DataFrame(tupArr, columns=['category', 'name'])
    counts = df.groupby('category', as_index=False).count()
    minCounts = counts[counts['name'] >= minCount]

    filteredDF = df.merge(minCounts['category'], on='category')
    np.savetxt(outP, filteredDF, fmt='%s')

