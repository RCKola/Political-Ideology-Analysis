from partisannet import get_topics, get_dataloaders

def show_topics():
    """Demonstrate topic modeling on the LibCon dataset using BERTopic."""
    data_dict = get_dataloaders(dataset="LibCon", batch_size=1000)
    loader = data_dict['test']
    batch = next(iter(loader)) 

    docs = [text for text in batch['text']]
    topic_model, topics, probs = get_topics(docs, remove_stopwords=True)

    print("Topic Info:", topic_model.get_topic_info())
    print("Topic 0:", topic_model.get_topic(0))
    print("Topic labels:", topic_model.generate_topic_labels(nr_words=1))

if __name__ == "__main__":
    show_topics()