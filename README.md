
# Fake News Generator

## Overview

The Fake News Generator is a web-based application designed to demonstrate the creation of fake news articles. The project provides functionality for text generation, training data input, and evaluation of generated content. This document explains how the project works, showcases the user interface, and guides users on how to use the application.

## Features

### 1. Admin and Client Authentication

- **Admin Login & Registration**: Secure login for administrators to access advanced features.
- **Client Login & Registration**: Separate access for clients to interact with the generator.
- **Password Recovery and Reset**: Options for recovering and resetting forgotten passwords.

### 2. Text Generation

Admins can generate text based on prompts. The application produces a sample text, which can be saved in the database for future training or modification.

### 3. Model Training

The application allows admins to upload text files to train the model further. Each training file has a size limit of 200MB and must be in `.txt` format.

### 4. Fake News Recognition Test

Users can take a quiz to identify fake news. The system provides sample news articles, and users must label them as "True" or "False."

## Generated Text Example

### Original Output:

```
Noua lege a pensiilor, adoptata de Senat [SEP] Senatul a adoptat, marti, o lege care prevede ca pensiile din 2024 sa fie majorate cu 13,8% din PIB, in timp ce alte tari au adoptat aceeasi lege. [SEP] Astfel, in contextul in care majorarea pensiilor este in vigoare, Senatul a adoptat, marti, o lege care prevede ca pensiile din 2024 sa fie majorate cu 13,8% din PIB, in timp ce alte tari au adoptat aceeasi lege.Astfel, in contextul in care majorarea pensiilor din 2024 sa fie majorate cu 13,8% din PIB, in timp cealate, Senatul a adoptat, marti o lege care prevede ca pensi ca pensiile din 2024 sa fie majorarea pensi pensiilor din 2024 sa fie majorate cu 13,8% din PIB este inseamna pensia pensia pensie, insemn pensia pensia pensia din 2022, iar pensia din 20% din PIB a pensi insemnia pensia din 20% din PIB a pensi pensi pensi pensi
```

### Edited for Clarity by human interfiere: 

"Noua lege a pensiilor, adoptată de Senat, prevede că pensiile din 2024 vor fi majorate cu 13,8% din PIB. Aceasta se aliniază cu legislații similare adoptate în alte țări. În acest context, Senatul a aprobat legea marți, asigurând o ajustare semnificativă a fondurilor pentru pensii."

This example shows how the generated text can be refined to create coherent and plausible content.

## How to Use

1. **Login/Registration**: Access the platform as an Admin or Client.
2. **Generate Text**: Input a prompt and generate fake news content.
3. **Train the Model**: Upload `.txt` files to improve the model’s performance.
4. **Test Recognition**: Participate in the fake news detection quiz.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RIAdrian/BS_Final_Project_Fake_News_Generator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BS_Final_Project_Fake_News_Generator
   ```
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the development server:
   ```bash
   npm start
   ```

## Screenshots

Below are the screenshots showcasing the application:

### Login Page

![Login Page](https://github.com/RIAdrian/BS_Final_Project_Fake_News_Generator/tree/main/Imagini/login.png)

### Text Generation

![Text Generation](https://github.com/RIAdrian/BS_Final_Project_Fake_News_Generator/tree/main/Imagini/Generate.png)

### Model Training

![Model Training](https://github.com/RIAdrian/BS_Final_Project_Fake_News_Generator/tree/main/Imagini/learning.png)

### Fake News Recognition Test

![Fake News Recognition Test](https://github.com/RIAdrian/BS_Final_Project_Fake_News_Generator/tree/main/Imagini/test.png)

### Fake News Recognition Test Results

![Fake News Recognition Test Results](https://github.com/RIAdrian/BS_Final_Project_Fake_News_Generator/tree/main/Imagini/results_test.png)

## Future Improvements

- Enhance the quality of generated text.
- Add more robust fake news detection mechanisms.
- Expand the dataset for training.

# Instructions for Training the GPT-2 Model 

Follow these steps to train a GPT-2 model using the provided code.

## Prerequisites

1. **Install Required Libraries**
   ```bash
   pip install transformers pandas scikit-learn
   ```

## Steps

1. **Load and Prepare Data**
   - Update the path to your data file.
   - Run the following code to load and preprocess your data:
     ```python
     import pandas as pd

     data_path = "path_to_your_data.xlsx"  # Update this path to your data file
     data = pd.read_excel(data_path)
     data['text'] = data['Titlu'] + " [SEP] " + data['Lead'] + " [SEP] " + data['Continut']
     texts = data['text'].tolist()
     ```

2. **Split Data into Training and Validation Sets**
   - Execute the following code to split your data and save it into text files:
     ```python
     from sklearn.model_selection import train_test_split

     train_texts, val_texts = train_test_split(texts, test_size=0.1)

     with open('train_texts.txt', 'w', encoding='utf-8') as f:
         f.write("\n".join(train_texts))
     with open('val_texts.txt', 'w', encoding='utf-8') as f:
         f.write("\n".join(val_texts))
     ```

3. **Load GPT-2 Tokenizer**
   - Run the following code to load the tokenizer:
     ```python
     from transformers import GPT2Tokenizer

     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
     ```

4. **Prepare the Datasets**
   - Create datasets for training and validation:
     ```python
     from transformers import TextDataset, DataCollatorForLanguageModeling

     train_dataset = TextDataset(
         tokenizer=tokenizer,
         file_path="train_texts.txt",
         block_size=128
     )

     val_dataset = TextDataset(
         tokenizer=tokenizer,
         file_path="val_texts.txt",
         block_size=128
     )

     data_collator = DataCollatorForLanguageModeling(
         tokenizer=tokenizer,
         mlm=False
     )
     ```

5. **Load GPT-2 Model**
   - Execute the following code to load the GPT-2 model:
     ```python
     from transformers import GPT2LMHeadModel

     model = GPT2LMHeadModel.from_pretrained('gpt2')
     ```

6. **Set Training Arguments**
   - Define training arguments by running:
     ```python
     from transformers import TrainingArguments

     training_args = TrainingArguments(
         output_dir='./model_output',  # Change this to your desired output directory
         overwrite_output_dir=True,
         num_train_epochs=15,
         per_device_train_batch_size=4,
         per_device_eval_batch_size=4,
         save_steps=10,000,
         save_total_limit=2,
         prediction_loss_only=True,
     )
     ```

7. **Train the Model**
   - Set up the Trainer and start training:
     ```python
     from transformers import Trainer

     trainer = Trainer(
         model=model,
         args=training_args,
         data_collator=data_collator,
         train_dataset=train_dataset,
         eval_dataset=val_dataset,
     )

     trainer.train()
     ```

8. **Save the Model**
   - Save the trained model and tokenizer:
     ```python
     model_path = './model_output'  # Update this to your desired path
     model.save_pretrained(model_path)
     tokenizer.save_pretrained(model_path)
     ```


# README for Main Application

## Overview
This application is designed to provide functionalities for user authentication, text generation, and model training using a pre-trained GPT-2 model. It includes features for both admin and client users.

## Prerequisites
- Python 3.x
- Install required libraries:
  ```bash
  pip install streamlit pandas scikit-learn transformers
  ```

## Running the Application
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run main.py
   ```

