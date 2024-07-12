def main():
    """Implement your assignment solution here"""
    import random 
    import string 
    import sys
    import os 
    """
    sample_sentence = "This is a sample sentence"
    tokenized_sentence = sample_sentence.split()

    with open('data/sample.csv','w') as f: 
        f.write(",".join(tokenized_sentence))
    
    """
    ### READ DATA ###

    data_path = sys.argv[1]
    neg_path = os.path.join(data_path,'neg.txt')
    pos_path = os.path.join(data_path,'pos.txt')

    with open(neg_path,'r') as f_neg:
        mylist_neg = f_neg.readlines()

    with open(pos_path,'r') as f_pos:
        mylist_pos = f_pos.readlines()
    
    data = mylist_neg + mylist_pos
    
    ### TOKENIZE ###
    
    def tokenize(file):
        tokenized_list = [line.strip().split() for line in file]
        return tokenized_list 
    
    data = tokenize(data)

    ### REMOVE SPECIAL CHARACTERS ###

    def no_specs(file): 
        specs = "!#$%&()*+/:,;.<=>@[\\]^`{|}~\t\n"
        spec_remove = str.maketrans("","",specs)
        clean = [[word.translate(spec_remove) for word in line] for line in file]
        return clean 
    
    data = no_specs(data)

    ## WRITE OUT.CSV ##

    def write_to_csv(base_dir, name, file_var):
        file_dir = os.path.join(base_dir, name)
        with open(file_dir, 'w') as f: 
            for line in file_var: 
                f.write(','.join(line) + "\n")
    
    write_to_csv(data_path, 'out.csv',data)

    ### TRAIN VAL TEST SPLIT ###
    
    def split(file): 
        
        train_size = int(0.8*len(file))
        val_size = int(0.1*len(file))

        random.shuffle(file)

        train = file[:train_size]
        val = file[train_size:train_size+val_size]
        test = file[train_size+val_size:]

        return train, val, test
    
    train, val, test = split(data)

    write_to_csv(data_path,'train.csv',train)
    write_to_csv(data_path,'val.csv',val)
    write_to_csv(data_path,'test.csv',test)

    ### REMOVE STOP WORDS ###

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
      
    def no_stops(file): 
        stopwords = load_stops()
        #print(f"Loaded stops: {stopwords}")
        clean_file = []
        for line in file: 
            cleaned_line = [word for word in line if word.lower() not in stopwords]
            #print(f"OG line: {line}")
            #print(f"Clean Line: {cleaned_line}")
            clean_file.append(cleaned_line)
        #print(f"Cleaned file: {file}")
        return clean_file 
    
    data_ns = no_stops(data)
    write_to_csv(data_path,'out_ns.csv',data_ns)

    train_ns, val_ns, test_ns = split(data_ns)
    write_to_csv(data_path,'train_ns.csv',train_ns)
    write_to_csv(data_path,'val_ns.csv',val_ns)
    write_to_csv(data_path,'test_ns.csv',test_ns)

if __name__ == "__main__":
    main()
