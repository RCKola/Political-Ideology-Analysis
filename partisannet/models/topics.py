from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


def get_topics(docs: list[str], num_topics: int | None = None, remove_stopwords: bool = False, embeddings=None) -> tuple[BERTopic, list[int], list[float]]:
    
    vectorizer = CountVectorizer(stop_words='english' if remove_stopwords else None)
    topic_model = BERTopic(nr_topics=num_topics, vectorizer_model=vectorizer)

    # 2. Add logic to use the embeddings if they exist
    if embeddings is not None:
        # BERTopic requires Numpy arrays, so we convert from PyTorch if needed
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        
        # Pass them into fit_transform
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    else:
        # Fallback to default (Standard SBERT) if no embeddings provided
        topics, probs = topic_model.fit_transform(docs)

    return topic_model, topics, probs