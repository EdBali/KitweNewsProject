import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

class NewsDataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        
        # Initialize tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Load the dataset with error handling
        try:
            print("Loading dataset...")
            self.df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully from {file_path}")
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
            self.df = None
        except pd.errors.EmptyDataError:
            print(f"Error: The file at {file_path} is empty.")
            self.df = None
        except Exception as e:
            print(f"Error: {e}")
            self.df = None

    def preview_data(self, n=5):
        """Display the first few rows of the dataset."""
        if self.df is not None:
            print(f"Previewing the first {n} rows of the dataset:")
            print(self.df.head(n))
        else:
            print("Error: No data available to preview.")

    def convert_date(self):
        """Convert 'Date' column to datetime format."""
        if self.df is not None and 'Date' in self.df.columns:
            try:
                print("Converting 'Date' column to datetime format...")
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                print("Date conversion successful.")
            except Exception as e:
                print(f"Error converting 'Date' column: {e}")
        else:
            print("Error: 'Date' column not found in dataset.")

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        if self.df is not None:
            print("Removing duplicate rows...")
            initial_rows = len(self.df)
            self.df.drop_duplicates(inplace=True)
            print(f"Removed {initial_rows - len(self.df)} duplicate rows.")
        else:
            print("Error: No data available to remove duplicates.")

    def fill_missing_values(self):
        """Fill missing values in the dataset."""
        if self.df is not None:
            print("Filling missing values...")
            self.df.fillna('Unknown', inplace=True)
            print("Missing values filled with 'Unknown'.")
        else:
            print("Error: No data available to fill missing values.")

    def normalize_text(self, text):
        """Lowercase and remove punctuation from text."""
        text = text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        return text

    def lemmatize_text(self, text):
        """Lemmatize the input text."""
        return ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def remove_stopwords(self, text):
        """Remove stopwords from the text."""
        return ' '.join([word for word in text.split() if word.lower() not in self.stop_words])

    def clean_text_columns(self):
        """Apply text cleaning to relevant columns."""
        if self.df is not None:
            print("Cleaning text columns ('Source', 'Category', 'Headline', 'Description')...")
            for column in ['Source', 'Category', 'Headline', 'Description']:
                if column in self.df.columns:
                    try:
                        self.df[column] = self.df[column].astype(str).apply(self.normalize_text) \
                                                              .apply(self.lemmatize_text) \
                                                              .apply(self.remove_stopwords)
                    except Exception as e:
                        print(f"Error cleaning {column} column: {e}")
                else:
                    print(f"Error: Column '{column}' not found in dataset.")
            print("Text cleaning completed.")
        else:
            print("Error: No data available to clean text columns.")

    def save_cleaned_data(self):
        """Save the cleaned dataset to a CSV file."""
        if self.df is not None:
            try:
                print(f"Saving cleaned data to {self.save_path}...")
                self.df.to_csv(self.save_path, index=False)
                print(f"Cleaned data saved to {self.save_path}")
            except Exception as e:
                print(f"Error saving cleaned data: {e}")
        else:
            print("Error: No data available to save.")

    def get_cleaned_data(self):
        """Returns the cleaned DataFrame."""
        return self.df

    def clean(self):
        """Main function to execute the entire cleaning process."""
        if self.df is not None:
            self.convert_date()
            self.remove_duplicates()
            self.fill_missing_values()
            self.clean_text_columns()
        else:
            print("Error: No data loaded. Cleaning process aborted.")