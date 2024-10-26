from data_cleaning import NewsDataCleaner
from text_categorizer import TextCategorizer
from fake_news_detector import FakeNewsDetector  # Assuming this is your detector class

def clean_data(file_path):
    cleaner = NewsDataCleaner(file_path)
    cleaner.clean()  # Run the full cleaning process
    return cleaner.get_cleaned_data()

def categorize_data(cleaned_data):
    categorizer = TextCategorizer(cleaned_data)
    return categorizer.categorize()  # Run the full categorization process

def detect_fake_news(categorized_data):
    detector = FakeNewsDetector(categorized_data)  # Assuming your detector takes the DataFrame
    return detector.determine_fake_news()  # Detect fake news

def save_data(data, output_path):
    data.to_csv(output_path, index=False)

def main(file_path, output_path):
    print("Starting data cleaning process...")
    cleaned_data = clean_data(file_path)
    
    if cleaned_data is not None:
        print("Data cleaning completed successfully.")

        print("Starting data categorization process...")
        categorized_data = categorize_data(cleaned_data)
        
        print("Starting fake news detection process...")
        final_data = detect_fake_news(categorized_data)

        print(f"Saving categorized data to {output_path}...")
        save_data(final_data, output_path)
        print(f"Categorized data saved to {output_path}.")
    else:
        print("Data cleaning failed. Categorization process aborted.")

if __name__ == "__main__":
    file_path = "News_Aggregator_Kitwe Data Collection - News_Aggregator_Kitwe Data Collection (1).csv"
    output_path = "final_data.csv"
    main(file_path, output_path)
