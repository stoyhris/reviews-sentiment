# Sentiment Classification of Amazon Reviews with Natural Language Processing 

## Introduction 

Understanding the voice-of-the-customer is at the core of many business decision. Through understanding how customers feel about their product, we can gain insights into important features or areas for improvement. This can drive marketing strategies as well as the product lifestyle, leading in a competitive advantage. 

However, with the amount of data available, it is challenging to derive accurate and objective insights. Natural language processing can be used to address this problem in a systematic way. This study proposes a neural-based classifier to determine the sentiment of a product review. Results achieve a 78% accuracy, which outperforms the naive benchmark by 26%.

## Data 

The analysis features Amazon's "Product Reviews" data-set. Collected in 2023, this large-scale dataset includes 54.51M users' written reviews of 48.19M products. The data has been further separated into positive reviews and negative reviews for training purposes. 

## Pre-processing 

In order to build data models, we must first pre-process our data. The goal of this step is twofold. First, most importantly, pre-processing allows us to turn data into a useable input that we can then use to train a model. Second, in natural language processing, different pre-processing steps can lead to different performance across models. With this in mind, we undertake and compare two different pre-processing pipelines.

To accomplish the first goal, we begin by tokenizing the data. Tokenization turns a sentence into a series of objects that can be interpreted individually. Next, we remove any special characters such as punctuation points and indents. Special characters do not indicate sentiment, but they do add complexity to the model. As such, removing them results in more standardized input data that will enable better performance. With these changes, we split the data into training, validation, and testing groups. From this pipeline, the sentence "I loved this product so much!" would become the collection of tokens: "I", "loved", "this", "product", "so", "much".

To accomplish the second goal, we augment the above set by also removing stop words. Stop words are common words such as "the". We hypothesize that, similarly to special characters, they add complexity without indicating sentiment. After this step, the sentence "I loved this product so much!" would become the collection of tokens: "loved", "product", "much".

Going forward, we consider the same training, validation, and testing splits of the two pre-processing pipelines.

## Baseline Classifier

A baseline classifier is the most simple model which can accomplish our task. We use it as a benchmark to see if the added complexity of a proposed classifier is paid off by its higher performance. In text classification, the most common baseline classifier is the Naive Bayes model. This draws on Bayes' Theorem from introductory probability: 

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

We can modify this theorem to fit out specific use case of predicting a class given a feature:

$$P(Class|Word) = \frac{P(Word|Class)P(Class)}{P(Word)}$$

The two probabilities $P(Class)$ and $P(Word)$ as well as the conditional probability $P(Word|Class)$ can be easily calculated from the data. Next, Naive Bayes relies on the assumption that each word in a sentence is independent. It follows that, for a sentence with $n$ words, we can use the following estimate: 

$$P(Class|Sentence) = \prod_{i=1}^nP(Class|Word_i)$$

Note that the independence assumption is not necessarily true in practice because language is inherently sequential. Further, if a word appears in a test document but not in any training document, the probability estimate for that word given the class will be 0, resulting in a zero probability for the entire sentence, leading to incorrect classification. As such, Laplace smoothing adds a small constant to the formula to mitigate this. 

* Let $N_{w_i}$ be the number of times word $w_i$ appears in documents of class $c$
* Let $N_c$ be the total number of words in documents of class $c$
* Let $V$ be the vocabulary size
* Let $\alpha$ be the smoothing parameter

We can reformulate our original probbility as follows: 

$$P(w_i|c) = \frac{N_{w_i} + \alpha}{N_c + \alpha V}$$

When training Naive Bayes models, we can choose how the model interprets the data. A unigram model treats each token separately. In contrast, a bigram model groups pairs of tokens together. For example, "I", "loved", "this", "product", "so", "much" would be read as "I loved", "this product", "so much". Finally, a unigram+bigram model combines both individual words and pairs of consecutive words to create a richer representation of the text. We train the Naive Bayes models using Python's Sklearn library. We consider models with/without stopwords that use unigrams, bigrams, or unigrams and bigrams, resulting in a total of six potential naive models. We tune the smoothing hyperparamter, $\alpha$, on the validation set and select the best naive baseline using the test set. 

| Multinomial Bayes Model Type        | Validation Accuracy | Test Accuracy |
|-------------------|---------------------|---------------|
| Unigram           | 0.5229125           | 0.521875      |
| Bigram           | 0.5003875           | 0.5002        |
| Unigram+bigram        | 0.5229625           | 0.5218875     |
| Unigram with No Stopwords        | 0.5264125           | 0.52335       |
| Bigram with No Stopwords         | 0.5023625           | 0.5016125     |
| Unigram+bigram with No Stopwords     | 0.5265875           | 0.524         |

### Stopword removal 

Across the models using unigram, bigram, and unigram+bigram, we see a slight improvement in performance when stop words are removed. We see this trend both in the validation accuracy and the test accuracy. This indicates that the performance may due to the stopword removal itself, and not due to better hyperparameter tuning on the validation set leading to overfitting on the test set. Note that our task is to classify sentences into positive or negative sentiment and that stopwords are general words which, in general, do not indicate a sentiment. As such, removing stopwords eliminates some noise during the training phase, allowing the model to focus on words which do indicate sentiment, and ultimately leading to better predictions on the validation and test sets.

### Unigrams, Bigrams, and Unigrams+Bigrams

Consider the four cases explored in the table above: validation, validation without stopwords, test, test without stopwords. Across all four cases, the unigram+bigram model performed best, followed closely by the unigram model, and the decisively worst model used bigrams. While the unigram+bigram and unigram models see comparable results, their difference is most pronounced in the testing phase without stopwords; I believe this is because the noise of stopwords and bias of hyperparameter tuning are removed, and so the unigram+bigram model shows itself as the highest performer. Note that a unigram model considers each word independently, a bigram considers each pair of adjacent words independently, and the unigram+bigram model considers both. In this case, each amazon review is relatively short, so a bigram model would get more confused assigning a positive/negative index to a pair of words, whereas a unigram model might assign strong positive/negative sentiment to individual terms such as "bad". Finally, the unigram+bigram model has the strength of the unigram model, but is able to perform slightly better perhaps because the bigram aspect of it helps to contextualize the broader meaning of the sentence and the unigram model keeps it from getting confused. 

In conclusion, our chosen benchmark classifier is a multinomial Naive Bayes model that uses unigrams and bigrams together on data with removed stopwords. The model training as well as an inference script that uses the model to predict the class of a given set of sentences are available under folder a2. 

## Word Vectorization 

Although Naive Bayes considers tokens and their frequency of appearance, there are more sophisticated approaches. Word2Vec is an unsupervised learning model which learns vector representations of words that capture semantic and syntactic information. For example, using word vectors we can successfully derive the following relationship: "King" - "Man" + "Woman" = "Queen".

First, we used Word2Vec to create a 300-dimensional vector space for every word in the dataset. Then, we explored that vector space by generating the most similar words to "good" and "bad". The code is available under folder a3, and the experiment is described below. 

The majority of words similar to "good" were positive, but there were a few negative words too, such as "bad", "poor", and "terrible", as well as one neutral word: "okay". Conversely, the majority of words similar to "bad" were negative, such as "horrible" and "terrible", but there were words that were not negative, such as "good" and "funny". I believe the reason this occurs is because 300-dimensional word vectors encode lots of information. Part of the information encoded is semantic, which is why "good" is close to lots of positive words, and "bad" is close to lots of negative words. However, another part of the information encoded may be contextual. For example, words of different sentiment can be used in the same context: "this toy is good" and "this toy is terrible", which also explains why ambiguous words such as "funny" may appear close to "bad", if they appear to have been used in a negative context in the training set. 

## Proposed Classifier

Fully connected feed-forward neural networks can be used to classify the sentiment of product reviews. A neural network begins with an input layer of vectors, which then undergo a series of linear and non-linear transformations in a series of hidden layers. Finally, there is an output layer which predicts the probability of each class and chooses the class with the highest probability. Training a neural network involves learning the best parameters for each linear transformation. Further, the choice of non-linear activation function can greatly affect the model's performance. 

Trained in PyTorch, our architecture involves the following steps, in combination with the Adam optimizer with L2-regularization: 
1. Data pooling: we create an average representation of all words in a sentence
2. An input layer with a linear transformation that our 300-dimensional sentence vector to the space of the hidden dimension
3. A non-linear activation function; we explored ReLU, Sigmoid, and Tanh
4. A dropout rate; this randomly sets a percentage of the weights to 0 so that the model does not overfit
5. A linear function that maps the space of the hidden dimension to the 2-dimensional output space
6. A Softmax which determines the class 

Once trained, a set of hyperparamters can be tuned on the validation set. 
* The hidden layer dimension is used to control the rate at which data changes. Note that our input layer is 300 but the output layer is 2; as such, the dimension of the hidden layer should be somewhat between the two dimensions to allow for a gradual transformation.
* The learning rate describes the rate at which the model iterates between solutions; if the rate is too low, the model will move in the right direction but fail to arrive at a good solution in time. In contrast, if the rate is too high, the model may oscillate between solutions, never arriving at the optimum.
* Inclusion of stopwords or not in our data
* The dropout percentage, which affects overfitting

Our best model used a hidden dimension of 125, a learning rate of 0.005, no stop words, and a 10% dropout rate with 10 epochs. We also included an inference script that attempted to classify the following sentences: 

* I love this product
* I hate this product, it does not work
* Why would anyone buy this product...
* Recommend to all my friends!
* i dont love it, but i like it

 Classifier Performance for each Activation Function

| Acitvation Function | Test Accuracy |
|------------|---------------|
| ReLU       | 78.198%       |
| Sigmoid    | 77.289%       |
| Tanh       | 77.546%       |

### Effect of Activation Function
It is worth noting that each of the functions performed very similarly. However, ReLU was clearly superior than Sigmoid and Tanh, which achieved almost identical performance. I believe this may be because Sigmoid and Tanh are both smooth functions, whereas ReLU is piecewise linear. Perhaps, when coupled with softmax, introducing non-linearity in a piecewise fashion results in a more robust network (indeed, when trying to classify an ambiguous sentence such as 'i dont love it, but i like it', the ReLU model was the only one that correctly identified it as positive). 

### Effect of L2-norm Regularization 
L2-norm regularization adds a penalty as the model complexity increases, which forces weights to be small instead of being zero. This prevents overfitting (increases bias but decreases variance). Indeed, when I trained a model without L2-norm regularization, it performed worse on the validation set, indicating that it overfit to the training set.

### Effect of Dropout Rate
Once the model learns the optimial weights for a given iteration, dropout forces a subset of them to randomly be set to 0 (unlike regularization, which incentivizes the model to learn weights closer to 0). This forces the model to diversify the network such that it is not overly dependent on any single neuron, leading to more robust predictions. However, dropout is a hyperparameter; setting it too low will result in potential overfitting, whereas setting it too high will result in the network not properly learning the relationships.

## Conclusion 

In conclusion, our proposed model can achieve a 78.2% classification accuracy, which greatly outperforms the 52.4% accuracy of the naive baseline model. Our contributions include reproducible results through training and inference scripts for each of the models, as well as the models themselves.
