# RNN-Based Text Classification and Machine Translation

This project demonstrates two key applications of **Recurrent Neural Networks (RNNs)**: emotion classification and machine translation. RNNs are powerful models for processing sequential data, such as text, where the order of words matters. In this project, we use RNNs, including LSTM (Long Short-Term Memory) networks, to handle tasks related to sentiment analysis and language translation.

---

## Project Overview
Recurrent Neural Networks (RNNs) are well-suited for tasks where context and sequence matter. However, traditional RNNs can suffer from vanishing and exploding gradient problems, which LSTMs and GRUs (Gated Recurrent Units) address by learning when to retain or forget information. 

In this project, we focus on two common use cases of RNNs:
1. **Emotion Classification** (many-to-one): Given a sentence, predict the emotion expressed in it.
2. **Machine Translation** (many-to-many): Translate text from one language to another.

---

## Objectives
- Build and train an RNN-based model for text emotion classification.
- Build and train an RNN-based machine translation model.
- Evaluate the models using accuracy, precision, recall, and other relevant metrics.
- Explore RNN variants like LSTMs to address issues of long-term dependencies in sequential data.

---

## Technologies Used
- **Python**: For programming the models and handling data.
- **Keras/TensorFlow**: For building and training RNN and LSTM models.
- **Pandas/Numpy**: For data preprocessing and manipulation.
- **Matplotlib/Seaborn**: For data visualization.

---

## Dataset
### 1. Emotion Classification
We use the **[Contextualized Affect Representations for Emotion Recognition](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)** from Kaggle. This dataset contains sentences labeled with one of six emotions (anger, joy, sadness, etc.).

### 2. Machine Translation
The dataset for machine translation consists of sentence pairs in two different languages. These pairs are used to train the RNN to translate text from one language to another. You can use the [OPUS dataset](https://opus.nlpl.eu/) or any other parallel corpus for translation tasks.

---

## Key Steps

1. **Data Preprocessing**:
   - For emotion classification, clean and tokenize the text data.
   - For machine translation, pair sentences in the source and target languages, tokenize and preprocess the data.
   - Apply padding and truncation to ensure the sequences are of uniform length.

2. **Modeling**:
   - For **Emotion Classification**, use a many-to-one RNN configuration to predict emotions.
   - For **Machine Translation**, use a many-to-many RNN configuration to translate sentences between two languages.
   - Implement **LSTM** layers to capture long-term dependencies in the text.

3. **Training and Evaluation**:
   - Train the models using the processed dataset.
   - Evaluate the models using accuracy, precision, recall, BLEU score (for machine translation), and visualize the results.

4. **Model Architecture**:
   - Use **embedding layers** to convert words into dense vectors.
   - Apply **LSTM layers** to process sequential text data.
   - Use a **Dense layer** with softmax activation for emotion classification and translation tasks.

---

## How to Use

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install tensorflow keras pandas numpy matplotlib seaborn
