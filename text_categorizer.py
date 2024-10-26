import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

class TextCategorizer:
    def __init__(self, data, n_neighbors=5, max_features=5000):
        self.data = data
        self.n_neighbors = n_neighbors
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # Define category keywords directly in the class
        self.categories_keywords = {
            'sports': ['football', 'soccer', 'basketball', 'tennis', 'cricket', 'olympics', 'athlete', 'sports'],
            'politics': ['government', 'election', 'politician', 'policy', 'parliament', 'minister', 'president', 'vote'],
            'education': ['school', 'university', 'education', 'college', 'students', 'learning', 'teacher', 'scholarship'],
            'health and wellness': ['health', 'hospital', 'doctor', 'wellness', 'mental health', 'fitness', 'medicine', 'disease'],
            'development': ['development', 'infrastructure', 'construction', 'road', 'bridge', 'building', 'urbanization'],
            'narcotics': ['narcotics', 'drug', 'cocaine', 'heroin', 'meth', 'drug trafficking', 'illegal drugs'],
            'fashion': ['fashion', 'clothing', 'designer', 'runway', 'model', 'style', 'apparel', 'trends'],
            'career': ['job', 'career', 'employment', 'opportunity', 'work', 'recruitment', 'hiring', 'position'],
            'local news': ['local', 'community', 'city', 'town', 'village', 'municipality', 'neighborhood', 'region'],
            'economy news': ['economy', 'economic', 'finance', 'market', 'stocks', 'currency', 'inflation', 'gdp'],
            'business news': ['business', 'company', 'corporation', 'entrepreneur', 'startup', 'industry', 'investment', 'profit']
        }

    def load_data(self):
        """Load and process initial data."""
        self.data['Description'] = self.data['Description'].astype(str)
        return self.data

    def prioritize_category(self, description):
        """Assign a single category based on highest keyword count."""
        keyword_count = {}
        for category, keywords in self.categories_keywords.items():
            count = sum(description.lower().count(keyword) for keyword in keywords)
            if count > 0:
                keyword_count[category] = count
        return max(keyword_count, key=keyword_count.get) if keyword_count else 'uncategorized'
    
    def assign_single_categories(self):
        """Apply single category based on keyword prioritization."""
        self.data['Single_Category'] = self.data['Description'].apply(self.prioritize_category)

    def train_knn_classifier(self):
        """Train the KNN model to predict categories for uncategorized entries."""
        categorized_df = self.data[self.data['Single_Category'] != 'uncategorized']
        uncategorized_df = self.data[self.data['Single_Category'] == 'uncategorized']
        
        # Train data
        X_train = categorized_df['Description']
        y_train = categorized_df['Single_Category']
        
        # Convert text to TF-IDF vectors
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train KNN classifier
        self.knn.fit(X_train_tfidf, y_train)
        
        # Predict uncategorized entries
        if not uncategorized_df.empty:
            X_test_tfidf = self.vectorizer.transform(uncategorized_df['Description'])
            y_pred = self.knn.predict(X_test_tfidf)
            self.data.loc[self.data['Single_Category'] == 'uncategorized', 'Single_Category'] = y_pred
    
    def categorize_description(self, description):
        """Assign multiple categories if needed."""
        categories = []
        for category, keywords in self.categories_keywords.items():
            if any(keyword in description.lower() for keyword in keywords):
                categories.append(category)
        return ', '.join(categories) if categories else 'uncategorized'
    
    def assign_multi_categories(self):
        """Assign multiple categories based on keyword presence."""
        self.data['Multi_Categories'] = self.data['Description'].apply(self.categorize_description)

    def get_categorized_data(self):
        """Return the categorized DataFrame."""
        return self.data

    def categorize(self):
        """Run all categorization steps in sequence, and replace 'Category' with 'Single_Category'."""
        self.load_data()
        self.assign_single_categories()
        self.train_knn_classifier()
        self.assign_multi_categories()
        
        # Replace 'Category' column with 'Single_Category' values and drop extra columns
        self.data['Category'] = self.data['Single_Category']
        self.data.drop(columns=['Single_Category', 'Multi_Categories'], inplace=True, errors='ignore')
        
        return self.get_categorized_data()
