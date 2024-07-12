def main():
    import os 
    import sys
    import random 
    from sklearn.feature_extraction.text import CountVectorizer 
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score
    import pickle
    
    def load_data(data_path):
        train_path = os.path.join(data_path, 'train.csv')
        val_path = os.path.join(data_path, 'val.csv')
        test_path = os.path.join(data_path, 'test.csv')

        with open(train_path, 'r') as f:
            train_data = [line.strip().split(',') for line in f.readlines()]

        with open(val_path, 'r') as f:
            val_data = [line.strip().split(',') for line in f.readlines()]

        with open(test_path, 'r') as f:
            test_data = [line.strip().split(',') for line in f.readlines()]

        return train_data, val_data, test_data
    
    def load_data_ns(data_path):
        train_path = os.path.join(data_path, 'train_ns.csv')
        val_path = os.path.join(data_path, 'val_ns.csv')
        test_path = os.path.join(data_path, 'test_ns.csv')

        with open(train_path, 'r') as f:
            train_data = [line.strip().split(',') for line in f.readlines()]

        with open(val_path, 'r') as f:
            val_data = [line.strip().split(',') for line in f.readlines()]

        with open(test_path, 'r') as f:
            test_data = [line.strip().split(',') for line in f.readlines()]

        return train_data, val_data, test_data
    
    def read_data(mylist):
        return [''.join(map(str,line[:-1])) for line in mylist], [line[-1] for line in mylist]

    data_path = sys.argv[1]
    
    train, val, test = load_data(data_path)
    X_train, y_train = read_data(train)
    X_val, y_val = read_data(val)
    X_test, y_test = read_data(test)

    train_ns, val_ns, test_ns = load_data_ns(data_path)
    X_train_ns, y_train_ns = read_data(train_ns)
    X_val_ns, y_val_ns = read_data(val_ns)
    X_test_ns, y_test_ns = read_data(test_ns)

    ### TRAIN MODELS NO STOPS ###
    def train_and_validate(X_train, y_train, X_val, y_val, X_test, y_test, ngram_range,alpha):

        vectorizer = CountVectorizer(ngram_range=ngram_range)
        X_train_vec = vectorizer.fit_transform(X_train) # Counts How many times a word appears in each line !
        X_val_vec = vectorizer.transform(X_val) #must be in the same space as X_trian, so no fit
        X_test_vec = vectorizer.transform(X_test)
        
        model = MultinomialNB(alpha=alpha) 
        model.fit(X_train_vec, y_train)
        y_val_pred = model.predict(X_val_vec)
        y_test_pred = model.predict(X_test_vec)

        val_accuracy = accuracy_score(y_val,y_val_pred)
        test_accuracy = accuracy_score(y_test,y_test_pred)

        return model, val_accuracy, test_accuracy, vectorizer

    ### TRAIN MODELS AND VALIDATE ON ONLY THE VALIDATION SET ###
    # Unigram w Stops 
    model_uni, val_accuracy, test_acc_uni, vectorizer = train_and_validate(X_train,y_train,X_val,y_val,X_test,y_test,(1,1),alpha=0.75)
    print("Unigram: ",val_accuracy)
    with open("a2/mnb_uni_vectorizer.pkl","wb") as f: 
        pickle.dump(vectorizer, f)
    # Bigram w Stops
    model_bi, val_accuracy, test_acc_bi, vectorizer = train_and_validate(X_train,y_train,X_val,y_val,X_test,y_test,(2,2),alpha=1)
    print("Bigram: ",val_accuracy)
    with open("a2/mnb_bi_vectorizer.pkl","wb") as f: 
        pickle.dump(vectorizer, f)
    # Unigram+Bigram w Stops
    model_unibi, val_accuracy, test_acc_unibi, vectorizer = train_and_validate(X_train,y_train,X_val,y_val,X_test,y_test,(1,2),alpha=0.75)
    print("Unigram+Bigram: ",val_accuracy)
    with open("a2/mnb_uni_bi_vectorizer.pkl","wb") as f: 
        pickle.dump(vectorizer, f)
    # Unigram No Stops 
    model_uni_ns, val_accuracy, test_acc_uni_ns, vectorizer = train_and_validate(X_train_ns,y_train_ns,X_val_ns,y_val_ns,X_test,y_test,(1,1),alpha=0.75)
    print("NS Unigram: ",val_accuracy)
    with open("a2/mnb_uni_ns_vectorizer.pkl","wb") as f: 
        pickle.dump(vectorizer, f)
    # Bigram No Stops
    model_bi_ns, val_accuracy, test_acc_bi_ns, vectorizer = train_and_validate(X_train_ns,y_train_ns,X_val_ns,y_val_ns,X_test,y_test,(2,2),alpha=0.25)
    print("NS Bigram: ",val_accuracy)
    with open("a2/mnb_bi_ns_vectorizer.pkl","wb") as f: 
        pickle.dump(vectorizer, f)
    # Unigram+Bigram No Stops
    model_unibi_ns, val_accuracy, test_acc_unibi_ns, vectorizer = train_and_validate(X_train_ns,y_train_ns,X_val_ns,y_val_ns,X_test,y_test,(1,2),alpha=0.25)
    print("NS Unigram+Bigram: ",val_accuracy)
    with open("a2/mnb_uni_bi_ns_vectorizer.pkl","wb") as f: 
        pickle.dump(vectorizer, f)
    
    ### EXAMINE ACCURACY ON THE TEST SET ###
    print("Test Unigram: ",test_acc_uni)
    print("Test Bigram: ",test_acc_bi)
    print("Test Unigram+Bigram: ",test_acc_unibi)
    print("Test NS Unigram: ",test_acc_uni_ns)
    print("Test NS Bigram: ",test_acc_bi_ns)
    print("Test NS Unigram+Bigram: ",test_acc_unibi_ns)

    ### SAVE MODELS ###

    with open('a2/mnb_uni.pkl','wb') as f:
        pickle.dump(model_uni,f)
    with open('a2/mnb_bi.pkl','wb') as f:
        pickle.dump(model_bi,f)
    with open('a2/mnb_uni_bi.pkl','wb') as f:
        pickle.dump(model_unibi,f)
    with open('a2/mnb_uni_ns.pkl','wb') as f:
        pickle.dump(model_uni_ns,f)
    with open('a2/mnb_bi_ns.pkl','wb') as f:
        pickle.dump(model_bi_ns,f)
    with open('a2/mnb_uni_bi_ns.pkl','wb') as f:
        pickle.dump(model_unibi_ns,f)

if __name__ == "__main__":
    main()
