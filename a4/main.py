import sys 
import csv 
import numpy as np 
import torch
import torch.nn as nn 
import os 
from torch.utils.data import DataLoader, TensorDataset 
import torch.optim as optim 
from gensim.models import Word2Vec 
import torch.nn.init as init


def load_data(file_path, w2v_model, max_entries = 10000000): 
    """ Load data and return something that a NN can read """
    embeddings = []
    labels = []
    entry_count = 0
    
    with open(file_path,'r') as file: 
        reader = csv.reader(file)
        for row in reader: 

            if entry_count >= max_entries:
                break

            tokens = row[:-1]
            label = int(row[-1])

            #print(tokens)
            #print(label)

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
            labels.append(label)
            #print(embeddings)
            entry_count += 1

        embeddings = np.array(embeddings) 
        labels = np.array(labels)
        
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return embeddings, labels

def create_model(input_dim, hidden_dim, output_dim, activation_function, dropout_rate):
    
    if activation_function == 'relu': 
        a_fn = nn.ReLU()
    elif activation_function == 'sigmoid': 
        a_fn = nn.Sigmoid()
    elif activation_function == 'tanh': 
        a_fn = nn.Tanh()
    
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        a_fn,
        nn.Dropout(dropout_rate), #not sure if this goes here or later
        nn.Linear(hidden_dim,output_dim),
        nn.LogSoftmax(dim=1)
    )

    for layer in model.children():
        if isinstance(layer,nn.Linear):
            init.xavier_uniform_(layer.weight)

    return model 

def train_and_evaluate(data_path, w2v_model_path, activation_function, dropout_rate, hidden_dim, learning_rate, epochs, stopwords):

    w2v_model = Word2Vec.load(w2v_model_path)
    
    if stopwords:
        # with stop words
        train_embeddings, train_labels = load_data(os.path.join(data_path, 'train.csv'), w2v_model)
        val_embeddings, val_labels = load_data(os.path.join(data_path, 'val.csv'), w2v_model)
        test_embeddings, test_labels = load_data(os.path.join(data_path, 'test.csv'), w2v_model)
    else: 
        # without stop words
        train_embeddings, train_labels = load_data(os.path.join(data_path, 'train_ns.csv'), w2v_model)
        val_embeddings, val_labels = load_data(os.path.join(data_path, 'val_ns.csv'), w2v_model)
        test_embeddings, test_labels = load_data(os.path.join(data_path, 'test_ns.csv'), w2v_model)

    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #batches for computational burden, shuffle to avoid learning patterns from order 
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = w2v_model.vector_size # placeholder
    output_dim = 2 #pos/neg 

    model = create_model(input_dim, hidden_dim, output_dim, activation_function, dropout_rate)
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5) # weight decay = l2 norm regularization 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    #print(f"Train labels: {train_labels}")
    #print(f"Train Embeddings shape: {train_embeddings.shape}")

    #print(f"Val labels: {val_labels}")
    #print(f"Val Embeddings shape: {val_embeddings.shape}")

    #print(f"Test labels: {test_labels}")
    #print(f"Test Embeddings shape: {test_embeddings.shape}")

    # Train model 
    print(f"Activation function {activation_function} raining start...")
    for epoch in range(epochs): 
        model.train() # training mode
        running_loss = 0.0
        iter = 0
        for inputs, labels in train_loader: 
            optimizer.zero_grad() # gradients from previous batch dont affect current batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            print(f'iteration {iter + 1} / {len(train_loader)}, training loss : {loss}')
            iter += 1
            #for name, param in model.named_parameters():
            #    if param.grad is not None:
            #        print(f'Layer {name} | Gradient norm: {param.grad.norm()}')
            # Gradients look fine in terms of size - neither blowing up nor shrinking
            
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Finished epoch {epoch}, training loss: {running_loss / len(train_loader)}")

    print(f'Done Training. Epoch {epoch}, activation function {activation_function}, training loss: {running_loss / len(train_loader)}')

    # Validation 
    model.eval() # evaluation mode
    val_loss = 0
    val_correct = 0 
    with torch.no_grad(): # makes it faster, we dont need gradients during evaluation 
        for inputs, labels in val_loader: 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() 
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
    val_accuracy = val_correct/len(val_dataset)
    print(f'Epoch {epoch}, activation function {activation_function}, validation accuracy {val_accuracy}')
    
    # Testing 
    model.eval() # evaluation mode
    test_loss = 0
    test_correct = 0 
    with torch.no_grad(): # makes it faster, we dont need gradients during evaluation 
        for inputs, labels in test_loader: 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() 
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = test_correct/len(test_dataset)

    print(f'Epoch {epoch}, activation function {activation_function}, test accuracy {test_accuracy}')
    
    if activation_function == 'relu':
        model_path = 'nn_relu.model'
    elif activation_function == 'sigmoid':
        model_path = 'nn_sigmoid.model'
    elif activation_function == 'tanh':
        model_path = 'nn_tanh.model'
    
    torch.save(model.state_dict(),model_path)

    return test_accuracy

def main():
    
    data_path = sys.argv[1]
    w2v_model_path = '/DATA1/shristov/assignments/a3/w2v.model'
    #activation_function = sys.argv[2]
    dropout_rate = 0.1 
    hidden_dim = int(125)
    learning_rate = 0.0005
    epochs = 10
    stopwords = False
    test_accuracies = []
    activation_functions = ['relu','tanh','sigmoid']
    for afn in activation_functions: 
        test_accuracy = train_and_evaluate(data_path,w2v_model_path, afn ,dropout_rate, hidden_dim, learning_rate, epochs, stopwords)
        test_accuracies.append(test_accuracy)

    accuracy_file = 'accuracy_file.txt'
    with open(accuracy_file, 'w') as f:
        for afn, accuracy in zip(activation_functions, test_accuracies): 
            f.write(f'{afn}: {accuracy}\n')

if __name__ == "__main__":
    main()

# Things that need tuning: hidden layer dimension, learning rate, stopwords/not, dropout rate