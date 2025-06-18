import os

import evaluate
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Gemma3Model,
    Trainer,
    TrainingArguments,
)

DEATH_DAYS = 1

os.environ["WANDB_PROJECT"] = "CS5555300"

model_name = "google/medgemma-4b-it"

# set random seed for reproducibility
torch.manual_seed(42)


def preprocess_dataset(examples):
    """Process images and prepare target values for model training"""
    # Process images with image processor
    inputs = image_processor(images=examples["image"], return_tensors="pt")

    # Convert labels to tensor
    target_values = []
    for i in range(len(examples["image"])):
        # event[all-cause mortality],time[all-cause mortality]
        target_value = int(
            examples["event[all-cause mortality]"][i]
            and examples["time[all-cause mortality]"][i] <= DEATH_DAYS
        )
        target_values.append(target_value)

    # Add processed images and targets to the dataset
    inputs["labels"] = torch.tensor(target_values).unsqueeze(
        1
    )  # Unsqueeze to make it 2D tensor

    return inputs


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


# Metrics computation function using PyTorch
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.from_numpy(predictions).sigmoid().round().long().numpy()
    return clf_metrics.compute(
        predictions=predictions,
        references=labels,
    )


def main():
    global image_processor

    # Load pre-trained image processor
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    # Load dataset fromCSV
    dataset = load_dataset("imagefolder", data_dir="./dataset")

    # Print dataset information
    print(f"Training set: {len(dataset['train'])} samples")
    print(f"Validation set: {len(dataset['validation'])} samples")

    # Set up dataset preprocessing
    dataset = dataset.map(
        preprocess_dataset,
        batched=True,
        batch_size=16,
        remove_columns=dataset["train"].column_names,
        num_proc=16,
    )

    gemma3_model = Gemma3Model.from_pretrained(model_name, device_map="cpu")

    # Initialize model
    model = AutoModelForImageClassification.from_pretrained(
        "google/siglip2-so400m-patch14-384",
        num_labels=1,
        ignore_mismatched_sizes=True,
        device_map="cpu",
    )

    model.vision_model = gemma3_model.vision_tower

    model.config.problem_type = "multi_label_classification"

    accelerator = Accelerator()
    model = model.to(accelerator.device)

    print(f"Model loaded on device: {accelerator.device}")

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(
            model.config.vision_config.hidden_size,
            model.config.vision_config.hidden_size // 2,
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(model.config.vision_config.hidden_size // 2, 1),
    )

    for module in model.classifier.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results_death",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=30,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        logging_dir="./logs",
        logging_steps=10,
        gradient_accumulation_steps=16,  # To handle larger effective batch sizes
        fp16=True,  # Use mixed precision for faster training
        dataloader_num_workers=4,  # Parallelize data loading
        label_names=["labels"],
        lr_scheduler_type="constant",
        optim="schedule_free_radam",
    )

    # Initialize trainer with compute_loss parameter
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model("./final_model_death")

    # Evaluate model
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nTraining complete! Model saved to './final_model'")


if __name__ == "__main__":
    main()
