from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def get_topics(docs: list[str], num_topics: int | None = None, remove_stopwords: bool = False) -> tuple[BERTopic, list[int], list[float]]:
    vectorizer = CountVectorizer(stop_words='english' if remove_stopwords else None)
    topic_model = BERTopic(nr_topics=num_topics, vectorizer_model=vectorizer)
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs