# Keep 3 letters because some are important
cat /workspace/datasets/fasttext/normalized_titles.txt | tr " " "\n" | grep "..." | sort | uniq -c | sort -nr | head -1004 | grep -oE '[^ ]+$' > /workspace/datasets/fasttext/top_words_v1.txt

python ./rmStopWords.py  # remove just real stop words

python ./genSynonyms.py  # creates /workspace/datasets/fasttext/synonyms.csv

docker cp /workspace/datasets/fasttext/synonyms.csv opensearch-node1:/usr/share/opensearch/config/synonyms.csv

# Modify bbuy_products.json

./index-data.sh -r -p /workspace/search_with_machine_learning_course/week2/conf/bbuy_products.json