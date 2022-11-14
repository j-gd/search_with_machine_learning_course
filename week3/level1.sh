pyenv activate search_with_ml

cat /workspace/datasets/fasttext/labeled_queries.txt | nl -w2 -s'> ' | egrep -i "Monster Pro"

cat /workspace/datasets/fasttext/labeled_queries_counts.txt | egrep __label__abcat0701001

python create_labeled_queries.py --min_queries 1000 
# > /workspace/datasets/fasttext/labeled_queries_enough_counts.txt
cut -d' ' -f1 /workspace/datasets/fasttext/labeled_queries.txt | sort | uniq | wc

# Task 2
shuf /workspace/datasets/fasttext/labeled_queries.txt > /workspace/datasets/fasttext/shuffled_labeled_queries.txt
head -50000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt > /workspace/datasets/fasttext/queries.train
tail -10000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt > /workspace/datasets/fasttext/queries.test

 ~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/queries.train -output /workspace/datasets/fasttext/classifier -lr 0.7 -epoch 5
 ~/fastText-0.9.2/fasttext test /workspace/datasets/fasttext/classifier.bin /workspace/datasets/fasttext/queries.test