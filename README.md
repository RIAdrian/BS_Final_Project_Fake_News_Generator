
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

