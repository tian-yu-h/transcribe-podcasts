# Example: reading segmented transcript from file
segments = []
with open("cleaned_combined_topics.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # We skip empty lines (those just used as separators)
        if line:
            segments.append(line)

print(f"Number of segments: {len(segments)}")


#########################
### 1 LDA with Gensim ###
#########################
import re
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords

cleaned_segments = [remove_stopwords(seg) for seg in segments]
tokenized_segments = [seg.split() for seg in segments]

# Create a Gensim dictionary from the tokenized segments
dictionary = Dictionary(tokenized_segments)

# Filter out very rare or very common tokens (optional but recommended)
# For instance, remove words that appear in less than 2 documents or more than 70% of the documents
dictionary.filter_extremes(no_below=2, no_above=0.5)

# Convert each tokenized segment into a bag-of-words
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_segments]

print(f"Dictionary size: {len(dictionary)}")
print(f"Number of documents (segments): {len(corpus)}")

num_topics = 5  # Try 5 topics to start

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=10,        # Number of passes through the corpus during training
    alpha='auto'      # Let Gensim adjust alpha automatically
)

for i in range(num_topics):
    top_words = lda_model.print_topic(i, topn=10)
    print(f"Topic {i}: {top_words}\n")

################################
#### 2 NMF with scikit-learn ###
################################
# Print the resulting topics and top words

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# We use text segments (list of text segments)
vectorizer = TfidfVectorizer(min_df=2, max_df=0.7, stop_words='english')
X = vectorizer.fit_transform(segments)

num_topics = 5
nmf_model = NMF(n_components=num_topics, random_state=42)
W = nmf_model.fit_transform(X) #document-topic matrix
H = nmf_model.components_      #topic-word matrix

feature_names = vectorizer.get_feature_names_out()

for i, topic_weights in enumerate(H) :
    # get top 10 words for this topic
    top_indices = topic_weights.argsort()[::-1][:10]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"Topic {i}: {', '.join(top_words)}")

###################################################
### 3 Embedding-Based Topic Modeling (BERTopic) ###
###################################################
from bertopic import BERTopic

# based on 'segments' above
topic_model = BERTopic(language="english")
topics, _ = topic_model.fit_transform(segments)

# Print general topic info
topic_info = topic_model.get_topic_info()
print(topic_info.head())

# show top words for topic 0
print(topic_model.get_topic(0))