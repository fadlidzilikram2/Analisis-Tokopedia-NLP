import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Stopwords Indonesian (assume udah download)
stop_words = set(stopwords.words("indonesian"))

def preprocess_text(text):
    """
    Preprocessing teks Tokopedia:
    - Lowercase, hapus angka, punctuation
    - Tokenize, filter stopwords & short words
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Filter stopwords & panjang >2
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    return ' '.join(tokens)
