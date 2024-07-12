def main():
    import sys
    import os 
    import gensim 
    from gensim.models import Word2Vec 

    def read_data(folderpath):
        corpus = []
        for filename in ['pos.txt','neg.txt']:
            filepath = os.path.join(folderpath,filename)
            with open(filepath, 'r',encoding='utf-8') as f: 
                for line in f: 
                    words = gensim.utils.simple_preprocess(line)
                    corpus.append(words)
        return corpus 
    
    folder_path = sys.argv[1] 

    corpus = read_data(folder_path)

    lengths = []
    for i in range(len(corpus)):
        lengths.append(len(corpus[i]))
    
    #print(sum(lengths)/len(lengths)) average length is 14 - chose window size roughly a third of that


    model = Word2Vec(sentences = corpus, vector_size = 300, window=5, min_count = 1, workers=4)

    model.save("w2v.model")

if __name__ == "__main__":
    main()
