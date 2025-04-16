import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.optim import AdamW
import joblib

nltk.download('punkt')

# Step 1: Load and preprocess dataset
data = pd.read_csv('emotion_detection.csv')  # Replace with the path to your dataset
data = data.head(10)  # Use the first 1000 rows for training/testing

# Clean text (remove special characters, URLs, etc.)
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)  # Remove special characters
    text = text.lower().strip()
    return text

# Apply text cleaning
data['tweet'] = data['tweet'].apply(clean_text)
data['cause'] = data['cause'].apply(clean_text)

# Encode emotions into numeric labels
label_encoder = LabelEncoder()
data['emotion'] = label_encoder.fit_transform(data['emotion'])

# Split dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

# Step 2: Tokenizer and Input Preparation
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Store generated causes
generated_causes = []

class EmotionDataset(Dataset):
    def __init__(self, data, bert_tokenizer, bart_tokenizer, max_length=128):
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.max_length = max_length
        self.generated_causes = []
        self._generate_causes()
        
    def _generate_causes(self):
        for i in range(len(self.data)):
            tweet = self.data.iloc[i]['tweet']
            inputs = self.bart_tokenizer(tweet, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
            cause_ids = bart_model.generate(inputs.input_ids, max_length=50)
            generated_cause = bart_tokenizer.decode(cause_ids[0], skip_special_tokens=True)
            self.generated_causes.append(clean_text(generated_cause))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tweet = self.data.iloc[idx]['tweet']
        emotion = self.data.iloc[idx]['emotion']
        generated_cause = self.generated_causes[idx]
        
        combined_text = f"Tweet: {tweet} Cause: {generated_cause}"

        encoding = self.bert_tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {key: encoding[key].squeeze(0) for key in encoding}
        item['labels'] = torch.tensor(emotion)
        
        return item

train_dataset = EmotionDataset(train_data, bert_tokenizer, bart_tokenizer)
train_data['generated_cause'] = train_dataset.generated_causes
train_data.to_csv('updated_emotion_detection.csv', index=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Step 3: Fine-Tuning a Transformer for Emotion Classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

def train_model(model, train_loader, epochs=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

train_model(model, train_loader)

# Step 4: Evaluate the Model
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    target_names = label_encoder.classes_[:len(set(true_labels))]
    report = classification_report(true_labels, predictions, target_names=target_names)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")


test_dataset = EmotionDataset(test_data, bert_tokenizer, bart_tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)
evaluate_model(model, test_loader)

# Step 5: Save and Load the Model
model.save_pretrained("emotion_detection_model")
bert_tokenizer.save_pretrained("emotion_detection_model")
bart_tokenizer.save_pretrained("emotion_detection_model")
joblib.dump(label_encoder, 'label_encoder.pkl')  
