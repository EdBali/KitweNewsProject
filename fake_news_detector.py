import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np

class FakeNewsDetector:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.reputable_sources = [
            'bbc.com', 'reuters.com', 'nytimes.com', 'cnn.com', 'guardian.com', 'npr.org', 
            'forbes.com', 'bloomberg.com', 'washingtonpost.com', 'thetimes.co.uk', 
            'economist.com', 'wsj.com', 'cnbc.com'
        ]
        self.zambian_reputable_sources = self.reputable_sources + [
            'daily-mail.co.zm', 'times.co.zm', 'znbc.co.zm', 'flavaradioandtv.com', 
            'lusakatimes.com', 'kitwetimes.com'
        ]
        self.suspicious_domain_pattern = re.compile(r'\\.(info|lo|ru|cn|xyz|top|news|live|buzz|click|online)$')
        self.sensational_keywords = [
            'shocking', 'unbelievable', 'amazing', 'incredible', 'secret', 
            'exposed', 'you won’t believe', 'scandal', 'controversy'
        ]

    def check_source_credibility(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        if any(source in domain for source in self.zambian_reputable_sources):
            return -1
        elif self.suspicious_domain_pattern.search(domain):
            return 1
        return 0

    def detect_clickbait(self, headline):
        excessive_punctuation = len(re.findall(r'[!?.]{2,}', headline)) > 0
        all_caps = headline.isupper()
        provocative_words = any(word in headline.lower() for word in [
            'shocking', 'unbelievable', 'you won’t believe', 'secret', 
            'amazing', 'incredible'
        ])
        return 1 if excessive_punctuation or all_caps or provocative_words else 0

    def count_sensational_keywords(self, description):
        return sum(description.lower().count(word) for word in self.sensational_keywords)

    def apply_topic_modeling(self):
        # Fill NaN descriptions with empty strings
        self.data['Description'] = self.data['Description'].fillna("")
        
        count_vectorizer = CountVectorizer(max_features=300, stop_words='english')
        count_data = count_vectorizer.fit_transform(self.data['Description'].astype(str))
        
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(count_data)
        topic_distribution = lda.transform(count_data)
        
        # Ensure the dominant topics array matches the DataFrame length
        dominant_topics = topic_distribution.argmax(axis=1)
        if len(dominant_topics) < len(self.data):
            dominant_topics = np.pad(dominant_topics, (0, len(self.data) - len(dominant_topics)), constant_values=-1)
        
        return dominant_topics


    def get_sentiment_score(self, text):
        try:
            sentiment = TextBlob(text).sentiment
            return sentiment.polarity
        except Exception as e:
            return 0

    def check_mismatch_headline_description(self, row):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        combined_text = [row['Headline'], row['Description']]
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0] < 0.3

    def check_excessive_capitalization(self, text):
        words = text.split()
        capitalized_words = [word for word in words if word.isupper() and len(word) > 1]
        return len(capitalized_words) > 3

    def check_vague_author(self, author):
        vague_authors = ['admin', 'editor', 'newsroom', 'staff', 'unknown']
        return any(vague_name in author.lower() for vague_name in vague_authors)

    def count_suspicious_links(self, description):
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', description)
        return len(urls)

    def check_short_sensational_description(self, description):
        description_length = len(description)
        sensational_word_count = self.count_sensational_keywords(description)
        return description_length < 100 and sensational_word_count > 1
        

    def determine_fake_news(self):
        # Compute indicators and save directly to columns in `self.data`
        self.data['Source_Credibility'] = self.data['Link'].apply(self.check_source_credibility)
        self.data['Headline_Type'] = self.data['Headline'].apply(self.detect_clickbait)
        self.data['Sensational_Keyword_Count'] = self.data['Description'].apply(self.count_sensational_keywords)
        self.data['Dominant_Topic'] = self.apply_topic_modeling()  # Apply directly to DataFrame
        self.data['Sentiment_Score'] = self.data['Description'].apply(lambda x: self.get_sentiment_score(str(x)))
        self.data['Excessive_Capitalization'] = self.data['Headline'].apply(self.check_excessive_capitalization)
        self.data['Headline_Description_Mismatch'] = self.data.apply(self.check_mismatch_headline_description, axis=1)
        self.data['Vague_Author'] = self.data['Author'].apply(self.check_vague_author)
        self.data['Suspicious_Links_Count'] = self.data['Description'].apply(self.count_suspicious_links)
        self.data['Short_Sensational_Description'] = self.data['Description'].apply(self.check_short_sensational_description)

        # Use `apply` to consolidate indicators into 'Target_final'
        self.data['Target_final'] = self.data.apply(
            lambda row: self.enhanced_determine_fake_news(
                row,
                row['Source_Credibility'],
                row['Headline_Type'],
                row['Sensational_Keyword_Count'],
                row['Dominant_Topic'],
                row['Sentiment_Score'],
                row['Excessive_Capitalization'],
                row['Headline_Description_Mismatch'],
                row['Vague_Author'],
                row['Suspicious_Links_Count'],
                row['Short_Sensational_Description']
            ), axis=1
        )

        # Drop temporary indicator columns if they are no longer needed
        self.data.drop(columns=[
            'Source_Credibility', 'Headline_Type', 'Sensational_Keyword_Count', 'Dominant_Topic',
            'Sentiment_Score', 'Excessive_Capitalization', 'Headline_Description_Mismatch', 'Vague_Author',
            'Suspicious_Links_Count', 'Short_Sensational_Description'
        ], inplace=True)

        return self.data


    def enhanced_determine_fake_news(
        self, row, source_credibility, headline_type, sensational_keyword_count, dominant_topic,
        sentiment_score, excessive_capitalization, headline_description_mismatch, vague_author,
        suspicious_links_count, short_sensational_description
    ):
        fake_indicators = 0
        if source_credibility == 1:
            fake_indicators += 1
        if headline_type == 1:
            fake_indicators += 1
        if sensational_keyword_count > 2:
            fake_indicators += 1
        if dominant_topic in [1, 2]:
            fake_indicators += 1
        if abs(sentiment_score) > 0.5:
            fake_indicators += 1
        if excessive_capitalization:
            fake_indicators += 1
        if headline_description_mismatch:
            fake_indicators += 1
        if vague_author:
            fake_indicators += 1
        if suspicious_links_count > 2:
            fake_indicators += 1
        if short_sensational_description:
            fake_indicators += 1

        return 0 if fake_indicators >= 2 else 1
