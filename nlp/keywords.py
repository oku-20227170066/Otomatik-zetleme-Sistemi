from sklearn.feature_extraction.text import TfidfVectorizer

def get_keywords(text):
    vec = TfidfVectorizer(max_features=5)
    vec.fit([text])
    return vec.get_feature_names_out()