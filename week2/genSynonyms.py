import fasttext

threshold = 0.65

modelP = '/workspace/datasets/fasttext/title_model__minCnt_20.bin'
model = fasttext.load_model(modelP)

topWordsP = '/workspace/datasets/fasttext/top_words.txt'
synsP = '/workspace/datasets/fasttext/synonyms.csv'

with open(topWordsP) as file:
    topWords = file.readlines()
    # topWords = ['headphones', 'laptop', 'freezer', 'nintendo', 'whirlpool', 'kodak', 'ps2', 'razr', 'holiday', 'plasma', 'leather']  # test
    wdNsynsArr = []
    for word in topWords:
        word = word.strip()
        synArr = model.get_nearest_neighbors(word)
        filtSyns = [word]
        print(f"\nKeeping the following nearest neighbors of {word}:")
        for score, syn in synArr:
            if score > threshold:
                print(syn)
                filtSyns.append(syn)
        print("\n")
        synsTxt = ','.join(filtSyns)
        wdNsynsArr.append(synsTxt)
    wdNsyns = '\n'.join(wdNsynsArr)
    with open(synsP, 'w') as synFile:
        synFile.write(wdNsyns)
        print("Wrote " + synsP)
    

