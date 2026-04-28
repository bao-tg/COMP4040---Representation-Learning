import re
import pandas as pd
from tqdm import tqdm

def strip_html(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'<.*?>', '', text)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove extra whitespaces
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

def preprocess_data(df, min_length=10, max_tokens=256):
    """
    Full cleaning pipeline as specified in the project plan.
    """
    print("Cleaning HTML and whitespaces...")
    df['cleaned_text'] = df['text'].progress_apply(strip_html).progress_apply(clean_text)
    
    print("Handling duplicates...")
    df = df.drop_duplicates(subset=['user_id', 'parent_asin', 'cleaned_text'])
    
    print(f"Filtering reviews shorter than {min_length} characters...")
    df = df[df['cleaned_text'].str.len() >= min_length].copy()
    
    print(f"Truncating to {max_tokens} tokens...")
    # Simple whitespace tokenization for truncation
    def truncate(text):
        tokens = text.split()
        if len(tokens) > max_tokens:
            return " ".join(tokens[:max_tokens])
        return text
    
    df['cleaned_text'] = df['cleaned_text'].progress_apply(truncate)
    
    return df
