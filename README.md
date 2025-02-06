# 📝 Text Generation Using Bidirectional LSTM

🚀 This project implements a **Text Generation model** using a **Bidirectional LSTM** trained on **Shakespeare's text corpus**. The model learns text patterns and generates sequences based on a given seed phrase.

---

## 📌 Table of Contents

- [📖 Introduction](#-introduction)
- [📂 Dataset](#-dataset)
- [🛠️ Installation](#️-installation)
- [🚀 Model Training](#-model-training)
- [🤖 Text Generation](#-text-generation)
- [📢 Notes](#-notes)
- [📝 License](#-license)

---

## 📖 Introduction

This project focuses on **generating text sequences** using a **Bidirectional LSTM** model. The model is trained on Shakespearean text and can generate similar styled text based on a given seed phrase.

---

## 📂 Dataset

The dataset consists of Shakespearean text, preprocessed into tokenized sequences for training.

---

## 🛠️ Installation

Make sure you have Python installed. Then, clone this repository and install the dependencies:

```bash
git clone https://github.com/harshhmaniya/Text-Generation-Using-Bidirectional-LSTM.git
cd Text-Generation-Using-Bidirectional-LSTM
pip install -r requirements.txt
```

---

## 🚀 Model Training  

Run the Jupyter Notebook to preprocess the dataset and train the model:  

1️⃣ Open the Jupyter Notebook:  
```bash
jupyter notebook text_generation_lstm.ipynb
```

2️⃣ Run all the cells to:  
   - Load and preprocess the Shakespeare text corpus  
   - Tokenize the text and create training sequences  
   - Define and train the **Bidirectional LSTM** model  
   - Save the trained model as **`text_gen_shake.keras`**  

---

## 🤖 Text Generation

### 1️⃣ Load the Model

```python
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("text_gen_shake.keras")
```

### 2️⃣ Load Tokenizer

```python
import pickle

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
```

### 3️⃣ Generate Text

```python
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def generate_text(seed_text, model, tokenizer, max_sequence_len, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict the next word
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Convert index to word
        output_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + output_word

    return seed_text

# Example Usage
generated_text = generate_text("Shall I compare thee", model, tokenizer, max_sequence_len=50, next_words=20)
print(generated_text)
```

---

## 📢 Notes

- Ensure TensorFlow, NumPy, and Keras are installed before running the code.
- The tokenizer used during training is essential for making predictions.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/harshhmaniya/Text-Generation-Using-Bidirectional-LSTM/blob/main/LICENSE) file for details.


## Author
- **Harsh Maniya**  
- [LinkedIn](https://linkedin.com/in/harsh-maniya)
- [GitHub](https://github.com/harshhmaniya)
