import re
from transformers import pipeline
import os
import joblib


class RejectionCategory:
    def __init__(self, model_path='data/bart_model/bart_zero_classifier'):
        if model_path and os.path.exists(model_path):
            self.classifier = joblib.load(model_path)
        else:
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        self.categories = [
            "Medical Justification Denials",
            "Contractual or Financial Benefit Denials",
            "Information Errors or Omissions Denials"]

    def save_classifier(self, model_path):
        joblib.dump(self.classifier, model_path)

    def classify_message(self, message):
        result = self.classifier(message, self.categories)
        labels = []
        for i in range(len(result)):
            most_probable_label = result[i]['labels'][0]
            labels.append(most_probable_label)
        return labels

    def preprocess_messages(self, messages):
        cleaned_messages = []
        nphies_code_pattern = r'[A-Z]{2}-\d+-\d+'

        for message in messages:
            message = message.strip().strip("'''")

            message = re.sub(f"({nphies_code_pattern})", r' \1 ', message)
            parts = message.split()

            unique_parts = []
            seen = set()
            for part in parts:
                if part not in seen:
                    seen.add(part)
                    unique_parts.append(part)

            cleaned_message = ' '.join(unique_parts)
            cleaned_message = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_message)
            cleaned_messages.append(cleaned_message.strip())

        return cleaned_messages