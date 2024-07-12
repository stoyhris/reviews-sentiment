
import gensim 
import sys

def main(): 

    input_file = sys.argv[1]

    def read_words(filepath):
        with open(filepath,'r',encoding='utf-8') as f: 
            return [line.strip() for line in f]
    
    words = read_words(input_file)

    model_name = "w2v.model"
    model = gensim.models.Word2Vec.load(model_name)

    def get_similar_words(model, word): 
        return model.wv.most_similar(word, topn=20)
    
    for word in words: 
        similar_words = get_similar_words(model,word)
        print("Top-20 most similar words to ",word,": ")
        for similar_word, similarity in similar_words: 
            print(similar_word)
        print("\n")

if __name__ == "__main__":
    main()