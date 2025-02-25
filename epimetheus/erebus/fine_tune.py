import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def fine_tune_model():
    # load tokenizer annd model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Create dataset from the logs file
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="argocd_logs.txt", # the logs file MUST be there
        block_size=128
    )

    # prepare data collator (no MLM for GPT-2)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Set Training Arguments
    training_args = TrainingArguments(
        output_dir="./gpt_argocd_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limits=2,
        prediction_loss_only=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./gpt_argocd_model")
    tokenizer.save_pretrained("./gpt_argocd_model")
    print("Model fine-tuning complete and saved in './gpt_argocd_model'.")


if __name__ == "__main__":
    fine_tune_model()

# Run this script to fine-tune the model on your logs
# Fine-tuning a GPT-like model may require a machine with a GPU, 
# and you may have to adjust hyperparameters to suit your dataset