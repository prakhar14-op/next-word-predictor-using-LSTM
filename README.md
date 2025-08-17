# Next Word Predictor using LSTM

This project demonstrates how to build and train a Long Short-Term Memory (LSTM) neural network to predict the next word in a sequence. The model is trained on a small corpus of text from a Frequently Asked Questions (FAQ) page.

## 📝 Overview

The primary goal of this project is to create a language model that can understand the context of a given sequence of words and predict the most likely word to follow. This is a fundamental task in Natural Language Processing (NLP) with applications in text autocompletion, chatbots, and language translation.

The process involves:
1.  **Text Preprocessing:** Cleaning and preparing the raw text data.
2.  **Tokenization:** Converting words into numerical representations.
3.  **Sequence Generation:** Creating input-output pairs for the model to learn from.
4.  **Model Building:** Designing a sequential model with Embedding and LSTM layers.
5.  **Training:** Training the model on the prepared data.
6.  **Prediction:** Using the trained model to generate new words based on an input seed text.

## 🛠️ Technology Stack

* **Python 3.x**
* **TensorFlow & Keras:** For building and training the neural network.
* **NumPy:** For numerical operations and data manipulation.

## 🧠 How It Works

### 1. Data Preparation

* **Tokenization:** The entire text corpus is processed using `tensorflow.keras.preprocessing.text.Tokenizer`. This builds a vocabulary of all unique words and assigns a unique integer index to each word.
* **Sequence Generation:** The model is trained to predict a word based on the words that came before it. To create training samples, we iterate through each sentence in the corpus and create n-gram sequences. For a sentence like "what is the course fee", the following sequences are generated:
    * `[what, is]`
    * `[what, is, the]`
    * `[what, is, the, course]`
    * `[what, is, the, course, fee]`
* **Padding:** Neural networks require inputs of a fixed length. Since our generated sequences have varying lengths, we use `pad_sequences` to pad them with zeros at the beginning (`padding='pre'`). All sequences are padded to the length of the longest sequence in the dataset.
* **Splitting Features and Labels:** For each padded sequence, the last word is treated as the label (y), and the words preceding it are the features (X).
    * **Example:** For the sequence `[0, 0, what, is, the, course]`,
        * `X` = `[0, 0, what, is, the]`
        * `y` = `[course]`
* **One-Hot Encoding:** The target variable `y` is categorical (one word out of the entire vocabulary). It is converted into a one-hot encoded vector using `to_categorical`.

### 2. Model Architecture

A `Sequential` model is built with the following layers:

1.  **Embedding Layer:** `Embedding(vocab_size, 100, input_length=max_len-1)`
    * This layer converts the integer-encoded word indices into dense vectors of a fixed size (100 in this case). It helps the model capture semantic relationships between words.
    * `vocab_size` is the total number of unique words plus one (for the padding token).
    * `input_length` is the length of the input sequences (`max_len - 1`).

2.  **LSTM Layers:** `LSTM(150, return_sequences=True)` followed by `LSTM(150)`
    * These are the core recurrent layers that process the sequence of word embeddings. They are capable of learning long-term dependencies in the data.
    * **Note:** When stacking LSTM layers, the first LSTM layer must have `return_sequences=True` so that it outputs a 3D tensor (the full sequence of hidden states) for the next LSTM layer to process.

3.  **Dense Layer:** `Dense(vocab_size, activation='softmax')`
    * This is the final output layer. It has `vocab_size` neurons, one for each word in the vocabulary.
    * The `softmax` activation function outputs a probability distribution over the entire vocabulary, indicating the likelihood of each word being the next word.

### 3. Training

* **Compilation:** The model is compiled using the `adam` optimizer and `categorical_crossentropy` as the loss function, which is suitable for multi-class classification problems.
* **Fitting:** The model is trained by calling `model.fit(X, y, epochs=100)`. It learns to minimize the loss by adjusting its internal weights over 100 epochs.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed. Then, install the required libraries:

```bash
pip install tensorflow numpy
```

## Installation & Usage
- Clone the repository:
  ```bash
  git clone [https://github.com/prakhar14-op/next-word-predictor-using-lstm.git](https://github.com/prakhar14-op/next-word-predictor-using-lstm.git)
  cd next-word-predictor-using-lstm
  ```
- Open the Jupyter Notebook (lstm_project.ipynb) and run the cells sequentially.
- The final cells in the notebook demonstrate how to use the trained model to predict the next 10 words for a given seed text.

## model architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(283, 100, input_length=56))
# Add return_sequences=True to the first LSTM layer
model.add(LSTM(150, return_sequences=True)) 
model.add(LSTM(150))
model.add(Dense(283, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```
mmmmmm
mmmmmmmmmmmmmmmmmmmmmmmmmmmmm
