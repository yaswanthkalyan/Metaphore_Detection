import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import re
import argparse

def identify_metaphor_sentence(text, metaphor_word):
    sentences = re.split(r'\.', text)
    
    # Iterate through sentences to find the one containing the metaphor word
    for sentence in sentences:
        if metaphor_word.lower() in sentence.lower():
            return sentence.strip()  # Return the first matching sentence
    
    # If no matching sentence is found, return None
    return None

def train(train_dataset_path):
    # Load the dataset
    df = pd.read_csv(train_dataset_path)

    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label_boolean'])

    # Replace metaphor IDs with words
    metaphor = {0: 'road', 1: 'candle', 2: 'light', 3: 'spice', 4: 'ride', 5: 'train', 6: 'boat'}
    df.replace({"metaphorID": metaphor}, inplace=True)

    # Replace the text with the first sentence containing the metaphor word
    df['text'] = df.apply(lambda x: identify_metaphor_sentence(x['text'], x['metaphorID']), axis=1)
    df = df.rename(columns={'metaphorID': 'metaphor_word'})

    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label_boolean'])

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Load pre-trained DistilBERT model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Tokenize and encode the text data, including metaphor word embeddings
    train_tokens = tokenizer(list(train_data["text"]), list(train_data["metaphor_word"]), padding=True, truncation=True, return_tensors="pt")

    # Create PyTorch DataLoader
    train_dataset = TensorDataset(
        train_tokens["input_ids"],
        train_tokens["attention_mask"],
        torch.tensor(list(train_data["label_boolean"].astype(int)))
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Fine-tune the DistilBERT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    losses = []
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = (
                input_ids.to(device),
            
                attention_mask.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        losses.append(loss)
        print(f"Epoch {epoch} Loss {loss}")


    model.save_pretrained("trained_model")

def main(train_dataset_path):
    train(train_dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DistilBERT model on the metaphor detection dataset.")
    parser.add_argument("train_dataset", type=str, help="Path to the training dataset CSV file.")
    args = parser.parse_args()
    
    main(args.train_dataset)
