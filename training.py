import os
from collections import Counter

import pandas as pd
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

def early_stopping_step(validation_loss, best_val_loss, counter):
    """Function that implements Early Stopping"""

    stop = False

    if validation_loss < best_val_loss:
        counter = 0
    else:
        counter += 1

    if counter >= 10:
        stop = True

    return counter, stop

def checkpointing(validation_loss, best_val_loss, model, optimizer, save_path):

    if validation_loss < best_val_loss:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
            },
            save_path,
        )
        print(f"Checkpoint saved with validation loss {validation_loss:.4f}")

def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    try:
        class_to_index = dataset.class_to_idx
    except AttributeError:
        class_to_index = dataset.dataset.class_to_idx
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()})


def train_epoch(model, optimizer, loss_fn, data_loader, device="cpu", max_grad_norm=1.0):
    training_loss = 0.0
    model.train()

    # Iterate over all batches in the training set to complete one epoch
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)
        loss = loss_fn(output, targets)

        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        training_loss += loss.data.item() * inputs.size(0)

    return training_loss / len(data_loader.dataset)


def predict(model, data_loader, device="cpu"):
    all_probs = torch.tensor([]).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)
            probs = torch.nn.functional.softmax(output, dim=1)
            all_probs = torch.cat((all_probs, probs), dim=0)

    return all_probs

def score(model, data_loader, loss_fn, device="cpu"):
    total_loss = 0
    total_correct = 0
    
    all_probs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Scoring", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)

            loss = loss_fn(output, targets)
            total_loss += loss.item() * inputs.size(0)

            # Get probabilities for AUC
            probs = torch.nn.functional.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Accuracy calculation
            correct = torch.eq(torch.argmax(output, dim=1), targets)
            total_correct += torch.sum(correct).item()

    # Flatten the lists into numpy arrays
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Calculate Metrics
    n_observations = len(data_loader.dataset) # More accurate than batch_size * len
    average_loss = total_loss / n_observations
    accuracy = total_correct / n_observations
    
    # Calculate ROC-AUC
    n_classes = all_probs.shape[1]
    if n_classes == 2:
        # Binary: pass probability of the positive class
        auc_score = roc_auc_score(all_targets, all_probs[:, 1])
    else:
        # Multiclass: One-vs-Rest macro average
        auc_score = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    
    return average_loss, accuracy, auc_score

def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=20,
    device="cpu",
    scheduler=None,
    checkpoint_path=None,
    early_stopping=False,
):
    # Track the model progress over epochs
    train_losses = []
    train_accuracies = []
    train_aucs = []
    val_losses = []
    val_accuracies = []
    val_aucs = []
    learning_rates = []

    # Create the trackers if needed for checkpointing and early stopping
    best_val_loss = float("inf")
    early_stopping_counter = 0

    print("Model evaluation before start of training...")
    # Test on training set
    train_loss, train_accuracy, train_auc = score(model, train_loader, loss_fn, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_aucs.append(train_auc)
    # Test on validation set
    validation_loss, validation_accuracy, validation_auc = score(model, val_loader, loss_fn, device)
    val_losses.append(validation_loss)
    val_accuracies.append(validation_accuracy)
    val_aucs.append(validation_auc)

    for epoch in range(1, epochs + 1):
        print("\n")
        print(f"Starting epoch {epoch}/{epochs}")

        # Train one epoch
        train_epoch(model, optimizer, loss_fn, train_loader, device)

        # Evaluate training results
        train_loss, train_accuracy, train_auc = score(model, train_loader, loss_fn, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_aucs.append(train_auc)

        # Test on validation set
        validation_loss, validation_accuracy, validation_auc = score(model, val_loader, loss_fn, device)
        val_losses.append(validation_loss)
        val_accuracies.append(validation_accuracy)
        val_aucs.append(validation_auc)

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy*100:.4f}%")
        print(f"Training AUC: {train_auc:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy*100:.4f}%")
        print(f"Validation AUC: {validation_auc:.4f}")

        # # Log the learning rate and have the scheduler adjust it
        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)
        if scheduler:
            # ReduceLROnPlateau expects the monitored metric (validation loss).
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss)
            else:
                scheduler.step()

        # Checkpointing saves the model if current model is better than best so far
        if checkpoint_path:
            checkpointing(
                validation_loss, best_val_loss, model, optimizer, checkpoint_path
            )

        # Early Stopping
        if early_stopping:
            early_stopping_counter, stop = early_stopping_step(
                validation_loss, best_val_loss, early_stopping_counter
            )
            if stop:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss

    return train_losses, val_losses, train_accuracies, val_accuracies, train_aucs, val_aucs