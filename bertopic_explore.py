#load clean segments from transcripts
segments = []
with open("segmented_transcript_all.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # We skip empty lines (those just used as separators)
        if line:
            segments.append(line)

print(f"Number of segments: {len(segments)}") 


# try to change a transformer model
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from umap import UMAP
import hdbscan

#load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

#custom vectorizer with lemmatization
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer =  super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemmatize_text(w) for w in analyzer(doc))
    

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = embedding_model.encode(segments)
vectorizer_model = LemmaCountVectorizer(stop_words = "english")

#create a KeyBERTInspired representation model
keybert_model = KeyBERTInspired()

#create a MaximaMarginalRelevance model for diversity
mmr_model = MaximalMarginalRelevance(diversity=0.8)

#Combine the two representation models
representation_model = [keybert_model, mmr_model]

umap_model = UMAP(n_neighbors=20,
                  n_components=5,
                  min_dist=0.1,
                  random_state=42)
#If you want to get many tiny clusters, try increasing n_neighbors.
#If you want more granular topics, decrease n_neighbors.
#If topics are overlapping too much, try lowering min_dist.

hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5,
                                min_samples=2,
                                metric='euclidean',
                                cluster_selection_method='eom')
#min_cluster_size: Minimum number of documents to form a cluster.
    #Higher = bigger clusters, fewer total topics.
    #Lower = smaller clusters, more total topics.
#min_samples: Controls how conservative the clustering is around outliers.
    #Higher = more outliers, fewer tight clusters.
    #Lower = fewer outliers, more documents forced into clusters.

# based on 'segments' above
topic_model = BERTopic(language="english", 
                       embedding_model=embedding_model,
                       vectorizer_model=vectorizer_model,
                       representation_model=representation_model,
                       umap_model=umap_model,
                       hdbscan_model=hdbscan_model)


topics, _ = topic_model.fit_transform(segments)
topic_info = topic_model.get_topic_info()
print(topic_info)

# show top words for topic 0
#print(topic_model.get_topic(0))

#topic_model.merge_topics(segments, topics, topics_to_merge=[1, 2]) #merge topic 1 and 2
topic_model.reduce_topics(segments, nr_topics=10)

# Generate nicer looking labels and set them in our model
topic_labels = topic_model.generate_topic_labels(nr_words=3,
                                                 topic_prefix=False,
                                                 word_length=15,
                                                 separator=", ")
topic_model.set_topic_labels(topic_labels)

# Manually selected some interesting topics to prevent information overload
topics_of_interest = [7, 8, 11, 12, 13, 15, 16, 18, 20, 22, 23, 25, 26, 28, 29]

# Visualize documents
topic_model.visualize_documents(
    segments, 
    embeddings=embeddings, 
    hide_annotations=False, 
    topics=topics_of_interest,
    custom_labels=True
)

