"""Loads the necessary libraries to run the script"""
import sys
from nltk.corpus import wordnet
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import webtext
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams, FreqDist
from collections import Counter
import time
start = time.time()
nltk.download('averaged_perceptron_tagger')

def load_data():
    """Loads data into the dataframe"""
    df = pd.read_csv(r'C:\Users\tikhonova\OneDrive\NLP\3monthemaildata.csv')
    df = df.applymap(str)
    return df

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def find_ngrams(input_list, n):
    """Groups words into ngrams"""
    return list(zip(*[input_list[i:] for i in range(n)]))


def tokenize(df, col, new_col, count_col,n):
    """Tokenizes, lemmetizes a given column, then saves as a new df"""
    # tokenizing and making it lower case
    df[new_col] = df[col].apply(lambda x: word_tokenize(x.lower()))
    df[new_col] = df[new_col].apply(lambda x: ','.join(map(str, x)))
    # removing punctuation
    df[new_col] = df[new_col].apply(lambda x: re.sub(r"[^a-zA-Z0-9]", " ", str(x)))
    df[new_col] = df[new_col].apply(lambda x: re.sub(r"[\d-]", " ", str(x)))
    # removing stop words
    additional_stopwords = list(df.Name)
   # stop = stopwords.words('english')  + additional_stopwords
   # df[new_col] = df[new_col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
     # lemmatizing
    lemmatizer = WordNetLemmatizer()
    #df[new_col] = df[new_col].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in x.split()]))
    df[new_col] = df[new_col].apply(lambda x: ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in x.split()]))
    df[count_col] = df[new_col].map(lambda x: find_ngrams(x.split(" "), 1))
    df[count_col] = df[count_col].apply(lambda x: Counter(x).most_common(n))

    return df

def create_ngrams(df, col, new_ngram_col, n_words, n_groups):
    """Groups words of a given column by n_words number, then counts most frequent occurences, returning a given number of n_groups """
    df[new_ngram_col] = df[col].map(lambda x: find_ngrams(x.split(" "), n_words))
    df[new_ngram_col] = df[new_ngram_col].apply(lambda x: Counter(x).most_common(n_groups))
    #creating a column for common phrases, grouping by 2,3,4 words
    return df

"""def group_by(df,cols,groupby_col,n):
   #Saves a new df with most common words (number = n) in a given groupby_col
    df = df.applymap(str)
    df = df.groupby(cols)[groupby_col].apply(lambda x: Counter(" ".join(x).split()).most_common(n))
    df = df.to_frame(name='email_text').reset_index()
    return df"""

def save_data():
    """Save the clean dataset into an excel file"""
    out_path = r"C:\Users\tikhonova\OneDrive\NLP\output.xlsx"
    writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
    facdf.to_excel(writer, sheet_name='Results',index=False)
    writer.save()


df = load_data()

df = tokenize(df,'Subject','Tokenized_Subject','Subject_Word_Count',2)
df = tokenize(df,'Body','Tokenized_Body',"Body_Word_Count",10)

df = create_ngrams(df, 'Tokenized_Body', 'Bigrams', 2, 5)
df = create_ngrams(df, 'Tokenized_Body', 'Threegrams', 3, 4)
df = create_ngrams(df, 'Tokenized_Body', 'Fourgrams', 4, 3)
df = create_ngrams(df, 'Tokenized_Body', 'Fivegrams', 5, 2)

save_data()
print('Duration: {} seconds'.format(time.time() - start))