import pandas as pd
import joblib
import gensim
from gensim import corpora
from gensim.models import LdaModel
from preprocessing import preprocess_text  # pake lo punya

# Load CSV
df = pd.read_csv('ulasan_tokopedia.csv')
df.columns = df.columns.str.lower()
df["review"] = df["review"].fillna("")  # kolom review ada[function]

# Preprocess
texts = df["review"].astype(str).apply(preprocess_text).tolist()
texts = [t for t in texts if len(t.strip()) > 0]  # filter kosong

print(f"Processed {len(texts)} reviews")

# Gensim dictionary & corpus (BoW)
tokenized_texts = [t.split() for t in texts]
dictionary = corpora.Dictionary(tokenized_texts)
dictionary.filter_extremes(no_below=2, no_above=0.5)  # clean
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

# Train LDA
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5,  # 5 topik buat data kecil
               passes=10, alpha='auto', eta='auto', random_state=42)

# Save
joblib.dump(lda, 'lda_model.pkl')
joblib.dump(dictionary, 'lda_dict.pkl')
joblib.dump(corpus, 'sample_corpus.pkl')  # bonus test

print("LDA model saved!")
print("\nSample topics:")
for idx, topic in lda.print_topics(num_words=8):
    print(f"Topik {idx}: {topic}")