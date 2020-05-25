import nltk 


import re
import numpy as np
import pandas as pd
from pprint import pprint
import os

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


papers = pd.read_csv('/home/honeycomb/Desktop/Minor2/Dataset/GeneralDataset.csv',encoding='ISO-8859-1')# Print head
print(papers)

papers.shape
print(papers.shape)
papers.SentimentValue.value_counts()
print(papers.SentimentValue.value_counts())

#%matplotlib inline
sns.set(style="darkgrid")
ax = sns.countplot(x='SentimentValue',  data=papers)
plt.show()


def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('[,\.!?@#*/]', '', text)
    return text_nopunct
papers['Text_Clean'] = papers['Tweet'].apply(lambda x: remove_punct(x))
print(papers)


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(papers.Text_Clean))
print("\n\n The cleaned data is: \n")
print(data_words)


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
# See trigram example
print("\n\n The example of the trigram dat is:\n")
print(trigram_mod[bigram_mod[data_words[0]]])
print(trigram_mod[bigram_mod[data_words[1]]])
print(trigram_mod[bigram_mod[data_words[2]]])
print(trigram_mod[bigram_mod[data_words[3]]])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
papers['Text_Final'] = [' '.join(sen) for sen in data_words_nostops]

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print("\n\n\n\n The example of lematized Data is: \n")
print(data_lemmatized[0:5])



from wordcloud import WordCloud
long_string = ','.join(list(papers['Text_Final'].values))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')# Generate a word cloud
wordcloud.generate(long_string)# Visualize the word cloud
wordcloud.to_image()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()




id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print("\n\n\n The example of corpus made is: \n")
print(corpus[0])

# Human readable format of corpus (term-frequency)
print("\n\n\n The example of human readable format of the corpus is")
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])


#helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(papers['Text_Final'])
plot_10_most_common_words(count_data, count_vectorizer)

#Word Matrix: 
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(papers['Text_Final'])

print("Word Matrix is: \n\n")
print(data_vectorized)
print("\n\n\n")


data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")



# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

num_topics=20
print("\n\n\n Total topics with word probabilities are: \n")
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

print("\n\n Percentage of a topic in a document \n\n")
print(lda_model[corpus[0]])
print("\n\n")

from gensim.test.utils import datapath

temp_file = datapath("model")
lda_model.save(temp_file)

 # Load a potentially pretrained model from disk.
lda_model = gensim.models.ldamodel.LdaModel.load(temp_file)

other_texts = [['He do not cook well ', 'why does he annoy me so much', 'this lockdown is getting on my nerve'],['Good activities', 'i am really feeling very energetic', 'she snores so much'],['he is so nice and kind', 'system panic', 'nice person']]
print(other_texts)
other_corpus = [id2word.doc2bow(text) for text in other_texts]
unseen_doc = other_corpus[0]
vector = lda_model[unseen_doc]

lda_model.update(other_corpus)
vector = lda_model[unseen_doc]

lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, alpha='auto', eval_every=5)  # learn asymmetric alpha from data
pprint(lda.print_topics())
print(vector)
"""
new_text = 'This Restaurant has very nice food and atmosphere'
tokens = [stemmer.stem(token) for token in tokenizer.tokenize(new_text.lower()) if token not in stopwords_fr]
lda_model[id2word.doc2bow(tokens)]
"""
# Compute Perplexity
print('\n\n\n The Perplexity of the Model is: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\n The Coherence Score of the model is: ', coherence_lda)
print("\n")



from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')
vis


mallet_path = '/home/honeycomb/Data/Codes/Data/mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

print("\n\n Unformatted Topics are: \n")
pprint(ldamallet.show_topics(formatted=False))
print("\n\n Formatted Topics are: \n")
pprint(ldamallet.show_topics(formatted=True))


coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score of MALLET LDA is : ', coherence_ldamallet)

#moving towards finding the most suitable number of topics to be taken for lda model
def compute_coherence_values_c_v(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def compute_coherence_values_u_mass(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list1, coherence_values1 = compute_coherence_values_c_v(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
model_list2, coherence_values2 = compute_coherence_values_u_mass(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
limit=40; start=2; step=6;

x = range(start, limit, step)
plt.plot(x, coherence_values1)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values for C_V"), loc='best')
plt.show()

y = range(start, limit, step)
plt.plot(y, coherence_values2)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values U_mass"), loc='best')
plt.show()

print("\n\n")
for m, cv in zip(x, coherence_values1):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

print("\n\n")
for m, umass in zip(y, coherence_values2):
    print("Num Topics =", m, " has Coherence Value of", round(umass, 4))

print("\n\nThe optimal model for c_v is: \n")
optimal_model = model_list1[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


print("\n\nThe optimal model for u_mass is: \n")
optimal_model = model_list2[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data_lemmatized)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
print("\n\n The most preferred topic is: \n")
print(df_dominant_topic.head(10))







# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
print("\n\n The topic distribution is: \n")
print(df_dominant_topics)
