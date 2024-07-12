# MSCI 641 Assignment 2 Report

## Results

| Model Type        | Validation Accuracy | Test Accuracy |
|-------------------|---------------------|---------------|
| mnb_uni           | 0.5229125           | 0.521875      |
| mnb_bi            | 0.5003875           | 0.5002        |
| mnb_uni_bi        | 0.5229625           | 0.5218875     |
| mnb_uni_ns        | 0.5264125           | 0.52335       |
| mnb_bi_ns         | 0.5023625           | 0.5016125     |
| mnb_uni_bi_ns     | 0.5265875           | 0.524         |

## Question 1: Stopwords vs No Stopwrods 

Across the models using unigram, bigram, and unigram+bigram, we see a slight improvement in performance when stop words are removed. We see this trend both in the validation accuracy and the test accuracy. This indicates that the performance may due to the stopword removal itself, and not due to better hyperparameter tuning on the validation set leading to overfitting on the test set. Note that our task is to classify sentences into positive or negative sentiment and that stopwords are general words which, in general, do not indicate a sentiment. As such, I believe that removing stopwords eliminates some noise during the training phase, allowing the model to focus on words which do indicate sentiment, and ultimately leading to better predictions on the validation and test sets.

## Question 2: Unigrams, Bigrams, or Unigrams+Bigrams

Consider the four cases explored tin the table above: validation, validation without stopwords, test, test without stopwords. Across all four cases, the unigram+bigram model performed best, followed closely by the unigram model, and the decisively worst model used bigrams. While the unigram+bigram and unigram models see comparable results, their difference is most pronounced in the testing phase without stopwords; I believe this is because the noise of stopwords and bias of hyperparameter tuning are removed, and so the unigram+bigram model shows itself as the highest performer. Note that a unigram model considers each word independently, a bigram considers each pair of adjacent words independently, and the unigram+bigram model considers both. In this case, each amazon review is relatively short, so a bigram model would get more confused assigning a positive/negative index to a pair of words, whereas a unigram model might assign strong positive/negative sentiment to individual terms such as "bad". Finally, the unigram+bigram model has the strength of the unigram model, but is able to perform slightly better perhaps because the bigram aspect of it helps to contextualize the broader meaning of the sentence and the unigram model keeps it from getting confused. 