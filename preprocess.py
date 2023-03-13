import streamlit as st
import pandas as pd
import numpy as np
# import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Define preprocessing function
import re
from nltk.corpus import stopwords
import nltk
import spacy
from bs4 import BeautifulSoup
import string
pd.options.mode.chained_assignment = None
from re import sub
import html as ihtml
from itertools import groupby
# from emot.emo_unicode import EMOTICONS_EMO
import emoji
# lemmatize words with spacy
import spacy
# nlp = spacy.load('en_core_web_sm-3.2.0')

def preprocess(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(r'', text)
    text = BeautifulSoup(ihtml.unescape(text)).text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)

    # Dictionary of English Contractions
    contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "i'd": "i would", "i'd've": "i would have","i'll": "i will", 
                     "i'll've": "i will have","i'm": "i am","i've": "i have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

    # Regular expression for finding contractions
    contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)
        
    text = expand_contractions(text)
    
    text = text.upper()
    chat_words_str = """
    AFAIK=As Far As I Know
    AFK=Away From Keyboard
    ASAP=As Soon As Possible
    ATK=At The Keyboard
    ATM=At The Moment
    A3=Anytime, Anywhere, Anyplace
    BAK=Back At Keyboard
    BBL=Be Back Later
    BBS=Be Back Soon
    BFN=Bye For Now
    B4N=Bye For Now
    BRB=Be Right Back
    BRT=Be Right There
    BTW=By The Way
    B4=Before
    B4N=Bye For Now
    CU=See You
    CUL8R=See You Later
    CYA=See You
    FAQ=Frequently Asked Questions
    FC=Fingers Crossed
    FWIW=For What It's Worth
    FYI=For Your Information
    GAL=Get A Life
    GG=Good Game
    GN=Good Night
    GMTA=Great Minds Think Alike
    GR8=Great!
    G9=Genius
    IC=I See
    ICQ=I Seek you (also a chat program)
    ILU=ILU: I Love You
    IMHO=In My Honest/Humble Opinion
    IMO=In My Opinion
    IOW=In Other Words
    IRL=In Real Life
    KISS=Keep It Simple, Stupid
    LDR=Long Distance Relationship
    LMAO=Laugh My A.. Off
    LOL=Laughing Out Loud
    LTNS=Long Time No See
    L8R=Later
    MTE=My Thoughts Exactly
    M8=Mate
    NRN=No Reply Necessary
    OIC=Oh I See
    PITA=Pain In The A..
    PRT=Party
    PRW=Parents Are Watching
    ROFL=Rolling On The Floor Laughing
    ROFLOL=Rolling On The Floor Laughing Out Loud
    ROTFLMAO=Rolling On The Floor Laughing My A.. Off
    SK8=Skate
    STATS=Your sex and age
    ASL=Age, Sex, Location
    THX=Thank You
    TTFN=Ta-Ta For Now!
    TTYL=Talk To You Later
    U=You
    U2=You Too
    U4E=Yours For Ever
    WB=Welcome Back
    WTF=What The F...
    WTG=Way To Go!
    WUF=Where Are You From?
    W8=Wait...
    7K=Sick:-D Laugher
    &=and
    """

    # Lowercase
    text = text.lower()

    text = ''.join(''.join(s)[:2] for _, s in groupby(text))

    # chat_words = {}
    # for line in chat_words_str.splitlines():
    #     if line:
    #         key, value = line.split('=')
    #         chat_words[key] = value

    # # Replace chat words with their full form
    # for key, value in chat_words.items():
    #     text = text.replace(key, value)

    text = text.replace("@[A-Za-z0-9]+","")

    # for emot in EMOTICONS_EMO:
    #     text = text.replace(emot, " ".join(EMOTICONS_EMO[emot].replace(",","").replace(":","").split()))
    
    def extract_emojis(text):
        new_text = []
        new_text.append(emoji.demojize(text, delimiters=("", "")))
        return " ".join(new_text)
    
    text = extract_emojis(text)

    text = text.replace("_", " ")

    text = text.replace("-", " ")

    text = text.replace("  ", " ")

    text = text.replace(r'\b\w\b', '').replace(r'\s+', ' ')

    # def lemmatize_words(text):
    #     return " ".join([token.lemma_ for token in nlp(text)])
    
    # text = lemmatize_words(text)

    text = text.replace('\d+', '')

    PUNCT_TO_REMOVE = string.punctuation
    def remove_punctuation(text):
        """custom function to remove the punctuation"""
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    
    text = remove_punctuation(text)



    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove('nor')
    all_stopwords.remove('no')
    all_stopwords.remove('but')
    all_stopwords.remove('too')
    all_stopwords.remove('very')
    all_stopwords.remove('just')
    all_stopwords.remove('don')
    all_stopwords.remove('doesn')
    all_stopwords.remove('didn')
    all_stopwords.remove('wasn')
    all_stopwords.remove('weren')
    all_stopwords.remove('isn')
    all_stopwords.remove('aren')
    all_stopwords.remove('haven')
    all_stopwords.remove('hasn')
    all_stopwords.remove('hadn')
    all_stopwords.remove('won')
    all_stopwords.remove('wouldn')
    all_stopwords.remove('shouldn')
    all_stopwords.remove('couldn')
    all_stopwords.remove('mustn')
    all_stopwords.remove('mightn')
    all_stopwords.remove('needn')
    all_stopwords.remove('shan')

    def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in all_stopwords])

    text = remove_stopwords(text)

    text = text.replace('[^\w\s]', '')

    # text = text.apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))






    ############################################################################################################

    # # Remove special characters
    # text = re.sub(r'[^\w\s]', '', text)

    # # Remove emoji
    # # text = emoji.get_emoji_regexp().sub(u'', text)

    # # Remove emoticon
    # text = re.sub(r'\:\)|\:\(|\:\-\)|\:\-\(', '', text)

    # # Remove comma
    # text = re.sub(r',', '', text)

    # # Remove punctuation
    # text = text.translate(str.maketrans('', '', string.punctuation))

    # # Convert n't to not
    # text = re.sub(r'n\'t', 'not', text)

    # # Tokenize text into words
    # words = word_tokenize(text)

    # # Remove stop words
    # stop_words = set(stopwords.words('english')) - set(['not'])
    # words = [w for w in words if not w in stop_words]

    # # Lemmatize words
    # lemmatizer = WordNetLemmatizer()
    # words = [lemmatizer.lemmatize(w, pos='v') for w in words]

    # # Join words back into text
    # text = ' '.join(words)

    return text
