import joblib

def load_models():
    tfidf = joblib.load("tfidf.pkl")
    nb = joblib.load("nb_model.pkl")
    lda = joblib.load("lda_model.pkl")
    dictionary = joblib.load("lda_dict.pkl")
    return tfidf, nb, lda, dictionary
