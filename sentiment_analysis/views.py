from django.shortcuts import render
from .forms import ReviewForm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Load the model and tokenizer
model = load_model('D:\workplace\sentiment_analysis_model\sentiment_model.h5')
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([])  # Re-fit tokenizer if needed

def index(request):
    form = ReviewForm()
    sentiment = None
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['text']
            review_seq = tokenizer.texts_to_sequences([review_text])
            review_pad = pad_sequences(review_seq, maxlen=200)
            prediction = model.predict(review_pad)
            sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
    else :
        form = ReviewForm()
    return render(request, 'sentiment_analysis/index.html', {'form': form, 'sentiment': sentiment})
