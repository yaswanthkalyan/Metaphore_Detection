import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

def identify_metaphor_sentence(text, metaphor_word):
    sentences = re.split(r'\.', text)
   
    # Iterate through sentences to find the one containing the metaphor word
    for sentence in sentences:
        if metaphor_word.lower() in sentence.lower():
            return sentence.strip()  # Return the first matching sentence
   
    # If no matching sentence is found, return None
    return None

def load_model(model_path):
    # Load the trained model from the specified path
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return model

def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label_boolean'])
   
    # Replace metaphor IDs with words
    metaphor = {0:'road', 1:'candle', 2:'light', 3:'spice', 4:'ride', 5:'train', 6:'boat'}
    df.replace({"metaphorID": metaphor}, inplace=True)
   
    # Replace the text with the first sentence containing the metaphor word
    df['text'] = df.apply(lambda x: identify_metaphor_sentence(x['text'], x['metaphorID']), axis=1)
    df = df.rename(columns={'metaphorID': 'metaphor_word'})
   
    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label_boolean'])
   
    return df

def main(test_file, model_path):
    # Load test data
    test_data = pd.read_csv(test_file)

    # Preprocess test data
    test_data = preprocess_data(test_data)

    # Load the trained model
    model = load_model(model_path)

    # Tokenize and encode the test data (similar to training process)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    test_tokens = tokenizer(list(test_data["text"]), list(test_data["metaphor_word"]), padding=True, truncation=True, return_tensors="pt")

    # Create PyTorch DataLoader
    test_dataset = TensorDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"]
    )

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Make predictions
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

    # Print or save the predictions as needed
    print(predictions)

if __name__ == "__main__":
    test_file = "SML_train.csv"

    model_path = "path/to/your/trained/model"
    main(test_file, model_path)