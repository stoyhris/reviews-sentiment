def main():
    import pickle 
    import sys
    from sklearn.feature_extraction.text import CountVectorizer 


    in_text_file = sys.argv[1]
    in_model_name = sys.argv[2]

    

    def load_model(model_name):
        if model_name == "mnb_uni":
            ngram_range = (1,1)
        elif model_name == 'mnb_bi':
            ngram_range = (2,2)
        elif model_name == 'mnb_uni_bi':
            ngram_range = (1,2)
        elif model_name == 'mnb_uni_ns':
            ngram_range = (1,1)
        elif model_name == 'mnb_bi_ns':
            ngram_range = (2,2)
        elif model_name == 'mnb_uni_bi_ns':
            ngram_range = (1,2)
        else: 
            print("Not a valid model name.")
        
        model_file = f"a2/{model_name}.pkl"
        vectorizer_file = f"a2/{model_name}_vectorizer.pkl"
        with open(model_file, 'rb') as f: 
            model = pickle.load(f)
        with open(vectorizer_file,'rb') as f: 
            vectorizer = pickle.load(f)

        return model, vectorizer 

    model, vectorizer = load_model(in_model_name)

    with open(in_text_file,'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    X_test_vec = vectorizer.transform(sentences)
    
    predictions = model.predict(X_test_vec)
    #print(predictions)
    label_mapping = {'0': 'negative','1': 'positive'}
    predictions = [label_mapping[pred] for pred in predictions]

    for i in range(len(sentences)):
        print(sentences[i], ": ",predictions[i])

if __name__ == "__main__":
    main()
