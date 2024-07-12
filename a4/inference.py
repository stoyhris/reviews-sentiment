import sys
import string
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec

# Read data, tokenize it 
def load_stops(): # source: https://gist.github.com/sebleier/554280#file-nltk-s-list-of-english-stopwords
        all_stops = """
        i
        me
        my
        myself
        we
        our
        ours
        ourselves
        you
        your
        yours
        yourself
        yourselves
        he
        him
        his
        himself
        she
        her
        hers
        herself
        it
        its
        itself
        they
        them
        their
        theirs
        themselves
        what
        which
        who
        whom
        this
        that
        these
        those
        am
        is
        are
        was
        were
        be
        been
        being
        have
        has
        had
        having
        do
        does
        did
        doing
        a
        an
        the
        and
        but
        if
        or
        because
        as
        until
        while
        of
        at
        by
        for
        with
        about
        against
        between
        into
        through
        during
        before
        after
        above
        below
        to
        from
        up
        down
        in
        out
        on
        off
        over
        under
        again
        further
        then
        once
        here
        there
        when
        where
        why
        how
        all
        any
        both
        each
        few
        more
        most
        other
        some
        such
        no
        nor
        not
        only
        own
        same
        so
        than
        too
        very
        s
        t
        can
        will
        just
        don
        should
        now
        """
        stopwords = set(all_stops.split())
        return stopwords
    
def load_and_tokenize(data_path):

    ### READ DATA ###

    with open(data_path,'r') as f:
        data = f.readlines()
    
    ### TOKENIZE ###
    tokenized_list = [line.strip().split() for line in data] 

    ### REMOVE SPECIAL CHARACTERS ###

    specs = "!#$%&()*+/:,;.<=>@[\\]^`{|}~\t\n"
    spec_remove = str.maketrans("","",specs)
    tokenized_cleaned = [[word.translate(spec_remove) for word in line] for line in tokenized_list]

    ### REMOVE STOP WORDS ###
 
    stopwords = load_stops()
        #print(f"Loaded stops: {stopwords}")
    clean_file = []
    for line in tokenized_cleaned: 
        cleaned_line = [word for word in line if word.lower() not in stopwords]
        #print(f"OG line: {line}")
        #print(f"Clean Line: {cleaned_line}")
        clean_file.append(cleaned_line)
    #print(f"Cleaned file: {file}")

    return clean_file

# Run through w2v 
def vectorize(data_ns, w2v_model): 
    """ Load data and return something that a NN can read """
    embeddings = []

    for tokens in data_ns:
        sentence_embeddings = [] 
        for token in tokens: 
            if token in w2v_model.wv: 
                sentence_embeddings.append(w2v_model.wv[token]) #w2v embeddings for each token
            else: 
                sentence_embeddings.append(np.zeros(w2v_model.vector_size)) #missing case
        if sentence_embeddings: 
            sentence_embedding_av = np.mean(sentence_embeddings, axis=0) #pool: average
        else: 
            sentence_embedding_av = np.zeros(w2v_model.vector_size) #missing case
        embeddings.append(sentence_embedding_av)

    embeddings = np.array(embeddings) 
    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    return embeddings

# Run input through ML model 

def classify_sentences(embeddings, activation_function):
    # recreate NN 
    input_dim = embeddings.shape[1]
    hidden_dim = 125
    output_dim = 2 

    if activation_function == 'relu': 
        a_fn = nn.ReLU()
        model_path = '/DATA1/shristov/assignments/a4/nn_relu.model'
    elif activation_function == 'sigmoid': 
        a_fn = nn.Sigmoid()
        model_path = '/DATA1/shristov/assignments/a4/nn_sigmoid.model'
    elif activation_function == 'tanh': 
        a_fn = nn.Tanh()
        model_path = '/DATA1/shristov/assignments/a4/nn_tanh.model'
    
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        a_fn,
        nn.Dropout(0.1), #not sure if this goes here or later
        nn.Linear(hidden_dim,output_dim),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(torch.load(model_path))
    model.eval() 

    # inference 
    with torch.no_grad():
        outputs = model(embeddings)
        _, predicted = torch.max(outputs, 1)
        predictions = predicted.tolist()

    return predictions 


def main():
    data_path = sys.argv[1]
    activation_function = sys.argv[2]
    w2v_model_path = '/DATA1/shristov/assignments/a3/w2v.model'
    w2v_model = Word2Vec.load(w2v_model_path)

    data_ns = load_and_tokenize(data_path) 
    #print(data_ns)
    inputs = vectorize(data_ns, w2v_model)
    predictions = classify_sentences(inputs, activation_function)

    for idx, prediction in enumerate(predictions):
        if prediction == 1: 
            classified = 'positive'
        elif prediction == 0: 
            classified = 'negative'
        print(f'Sentence {idx}: {classified}')

if __name__ == "__main__":
    main()