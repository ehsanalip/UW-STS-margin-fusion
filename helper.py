
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score,roc_curve, auc
from sklearn.model_selection import train_test_split
import ipywidgets as widgets
from IPython.display import display
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import seaborn as sns
import os
from monai.networks.nets import ResNet, resnet10, resnet50, resnet101
from monai.transforms import Compose, RandFlip, RandRotate90, ToTensor, Resize
from confidenceinterval import roc_auc_score as roc_auc_score_ci
import io
import PIL.Image
from sklearn.model_selection import KFold
from tqdm import tqdm

from datetime import datetime


def plot_to_tensor(fig):
    """Converts a matplotlib figure to a tensor for logging via TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image_tensor = to_tensor(image)  # Convert image to PyTorch tensor
    return image_tensor 

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax = plt.subplots()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndenumerate(cm):
        ax.text( i[1], i[0], format(j, fmt),  # Adjust how i and j are used
                 horizontalalignment="center",
                 #color="white" if cm[i[1], i[0]] > thresh else "black",
                )

    return fig
def plot_images(slice_idx):
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(4, 2, figsize=(8, 8))
    
    # Flatten the array of axes for easier access
    axs = axs.flatten()
    
    # Display each image in its respective subplot
    for i in range(8):
        axs[i].imshow(img[i][slice_idx], cmap='bone')
        axs[i].axis('off')  # Hide axes ticks
    
    plt.tight_layout()  # Adjust subplots to fit into the figure area nicely
    plt.show()
    
def count_model_parameters(model: nn.Module):
    """
    Counts the total and trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        dict: A dictionary containing total and trainable parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {"total_params": total_params, "trainable_params": trainable_params}

    
#### Modeling
def replicate_first_layer(state_dict_path, raw_model):
    state_dict = torch.load(state_dict_path)
    state_dict = {x.replace('module.',''): y for x, y in state_dict['state_dict'].items()}
    original_conv1_weight = state_dict['conv1.weight']  # Assuming this key exists and is correct
    replicated_conv1_weight = original_conv1_weight.repeat(1, 8//original_conv1_weight.size(1), 1, 1, 1)


    state_dict['conv1.weight'] = replicated_conv1_weight

    raw_model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore missing keys
    return raw_model
    
def sensivity_specifity_cutoff(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]


def train(model,optimizer,criterion, train_loader, epoch,writers, writer, fold, device = "cuda" if torch.cuda.is_available() else "cpu"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"F_{fold}, E_{epoch} Training")
    
    for i, (all_inputs) in progress_bar:
        case_ids = all_inputs['case_id']
        labels = all_inputs['label']
        #print(labels)
        inputs = all_inputs['input']
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(1)  # Ensure output is of correct shape

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.sigmoid(outputs).data > 0.5
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Store labels and predictions for AUROC calculation
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
        progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)
    
    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    
    # Calculate AUROC
    auroc = roc_auc_score(all_labels, all_predictions)
    threshold = sensivity_specifity_cutoff(all_labels, all_predictions)
    #print(threshold)
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    progress_bar.set_postfix(loss=avg_loss, acc=accuracy)
    
    # Log to TensorBoard
    writers[fold].add_scalar('training loss', avg_loss, epoch)
    writers[fold].add_scalar('training accuracy', accuracy, epoch)
    writers[fold].add_scalar('training AUROC', auroc, epoch)
    
    print('training loss:', avg_loss, ', training accuracy:', accuracy, ', training AUROC:', auroc)
    return {
        "loss": avg_loss,
        "acc": accuracy,
        "auroc": auroc,
        "threshold": threshold,
    }

def train_fusion(model,optimizer,criterion, train_loader, epoch,writers, writer, fold, device = "cuda" if torch.cuda.is_available() else "cpu"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"F_{fold}, E_{epoch} Training")
    
    for i, (all_inputs) in progress_bar:
        case_ids = all_inputs['case_id']
        labels = all_inputs['label']
        inputs = all_inputs['input']
        images = all_inputs['img']
        images, inputs, labels = images.to(device), inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, inputs)
        outputs = outputs.squeeze(1)  # Ensure output is of correct shape

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.sigmoid(outputs).data > 0.5
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Store labels and predictions for AUROC calculation
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
        progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)
    
    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    
    # Calculate AUROC
    auroc = roc_auc_score(all_labels, all_predictions)
    _, ci = roc_auc_score_ci(all_labels.squeeze().astype(int), all_predictions.squeeze(), confidence_level=0.95, average = "weighted")
    threshold = sensivity_specifity_cutoff(all_labels, all_predictions)
    #print(threshold)
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    progress_bar.set_postfix(loss=avg_loss, acc=accuracy)
    
    # Log to TensorBoard
    writers[fold].add_scalar('training loss', avg_loss, epoch)
    writers[fold].add_scalar('training accuracy', accuracy, epoch)
    writers[fold].add_scalar('training AUROC', auroc, epoch)
    
    print('training loss:', avg_loss, ', training accuracy:', accuracy, ', training AUROC:', auroc)
    return {
        "loss": avg_loss,
        "acc": accuracy,
        "auroc": auroc,
        "ci" : ci,
        "threshold": threshold,
    }


def validate(model,optimizer,criterion, val_loader, epoch,writers, writer, fold = 0, threshold = 0.5,device = "cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_preds_binary = []
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False)
    with torch.no_grad():
        for all_inputs in progress_bar:
            case_ids = all_inputs['case_id']
            labels = all_inputs['label']
            inputs = all_inputs['input']
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # Ensure output is of correct shape
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.sigmoid(outputs).data 
            preds_binary = preds> threshold
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_preds_binary.extend(preds_binary.cpu().numpy())
            progress_bar.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds_binary)
    auroc = roc_auc_score(all_labels, all_preds)
    _, ci = roc_auc_score_ci(np.array(all_labels, dtype=int), all_preds, confidence_level=0.95, average = "weighted")
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds_binary).ravel()
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Plot and convert confusion matrix
    cm_fig = plot_confusion_matrix(confusion_matrix(all_labels, all_preds_binary), classes=['Negative', 'Positive'])
    cm_image = plot_to_tensor(cm_fig)

    # Plot and convert ROC curve
    roc_fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
           title="Receiver Operating Characteristic",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    roc_image = plot_to_tensor(roc_fig)

    # Log to TensorBoard
    writers[fold].add_scalar('Validation loss', avg_val_loss, epoch)
    writers[fold].add_scalar('Validation accuracy', accuracy, epoch)
    writers[fold].add_scalar('Validation AUROC', roc_auc, epoch)
    writers[fold].add_scalar('validation sensitivity', sensitivity, epoch)
    writers[fold].add_scalar('validation specificity', specificity, epoch)
    writers[fold].add_image('Confusion Matrix', cm_image, epoch)
    writers[fold].add_image('ROC Curve', roc_image, epoch)

    
    # Update tqdm
    print(f"Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f}, AUROC: {auroc:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}")
    
    return {
        "loss": avg_val_loss,
        "acc" : accuracy,
        "auroc": auroc,
        "sen": sensitivity,
        "spec": specificity,
    }

def validate_fusion(model,optimizer,criterion, val_loader, epoch,writers, writer, fold = 0, threshold = 0.5,device = "cuda" if torch.cuda.is_available() else "cpu", print_plots = False):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_preds_binary = []
    all_case_ids = []
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False)
    with torch.no_grad():
        for all_inputs in progress_bar:
            case_ids = all_inputs['case_id']
            labels = all_inputs['label']
            inputs = all_inputs['input']
            images = all_inputs['img']
            images, inputs, labels = images.to(device), inputs.to(device), labels.to(device)
            outputs = model(images,inputs)
            outputs = outputs.squeeze(1)  # Ensure output is of correct shape
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.sigmoid(outputs).data 
            preds_binary = preds> threshold
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_preds_binary.extend(preds_binary.cpu().numpy())
            all_case_ids.extend(case_ids)
            progress_bar.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds_binary)
    auroc = roc_auc_score(all_labels, all_preds)
    _, ci = roc_auc_score_ci(np.array(all_labels, dtype=int), all_preds, confidence_level=0.95, average = "weighted")
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds_binary).ravel()
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Plot and convert confusion matrix
    cm_fig = plot_confusion_matrix(confusion_matrix(all_labels, all_preds_binary), classes=['Negative', 'Positive'])
    cm_image = plot_to_tensor(cm_fig)
    if print_plots:
        display(cm_fig)
        plt.show()
    # Plot and convert ROC curve
    roc_fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
           title="Receiver Operating Characteristic",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    roc_image = plot_to_tensor(roc_fig)
    if print_plots:
        display(roc_fig)
        plt.show()
    # Log to TensorBoard
    writers[fold].add_scalar('Validation loss', avg_val_loss, epoch)
    writers[fold].add_scalar('Validation accuracy', accuracy, epoch)
    writers[fold].add_scalar('Validation AUROC', roc_auc, epoch)
    writers[fold].add_scalar('validation sensitivity', sensitivity, epoch)
    writers[fold].add_scalar('validation specificity', specificity, epoch)
    writers[fold].add_image('Confusion Matrix', cm_image, epoch)
    writers[fold].add_image('ROC Curve', roc_image, epoch)

    
    # Update tqdm
    print(f"Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f}, AUROC: {auroc:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}")
    
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return {
        "loss": avg_val_loss,
        "acc" : accuracy,
        "auroc": auroc,
        "ci" : ci,
        "sen": sensitivity,
        "spec": specificity,
        "preds": list(zip(all_case_ids,all_preds))
    }

def train_model(config, checkpoint_dir=None,early_stopping = False, current_time = datetime.now().strftime("%Y%m%d-%H%M%S"), device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    logdir = f"Logs/{current_time}/{config['model']}_LR:{config['lr']}_NF:{config['n_fold']}_E:{config['epochs']}_B:{config['batch_size']}"  # Add a descriptive name to the path
    # Setup data for K-Folds
    k_folds = config['n_fold']
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # TensorBoard
    writers = [SummaryWriter(f"{logdir}/Fold_{i+1}") for i in range(k_folds)]
    writer = SummaryWriter(logdir)
    
    fold_val_losses = []
    fold_val_aucs = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(np.arange(len(train_dataset)))):
        
        # Initialize model and optimizer
        if config['model'] == 'resnet50':
            print('resnet50 selected.')
            model = resnet50(
                    pretrained=False,
                    spatial_dims=3,
                    n_input_channels=8,  
                    num_classes=1)
            model = replicate_first_layer(pre_resnet50_weights_path, model)
        elif config['model'] == 'resnet101':
            model = resnet101(
                    pretrained=False,
                    spatial_dims=3,
                    n_input_channels=8, 
                    num_classes=1)
        elif config['model'] == 'resnet10':
            print('resnet10 selected.')
            model = resnet10(
                    pretrained=False,
                    spatial_dims=3,
                    n_input_channels=8,  
                    num_classes=1)
            model = replicate_first_layer(pre_resnet10_weights_path, model)
            
        else:
            print("Model not selected properly.")
            
        model.to(device)    
        optimizer = optim.Adam(model.parameters(), lr=config["lr"],  weight_decay=config.get("weight_decay", 0))
        criterion = nn.BCEWithLogitsLoss()
        best_auc = 0
        best_loss = 0
        train_subsampler = Subset(train_dataset, train_ids)
        test_subsampler = Subset(train_dataset, test_ids)

        train_loader = DataLoader(train_subsampler, batch_size=int(config["batch_size"]), shuffle=True)
        val_loader = DataLoader(test_subsampler, batch_size=int(config["batch_size"]), shuffle=False)
        train_dict = {}
        val_dict = {}
        for epoch in range(config['epochs']): 
            train_dict = train(model, optimizer,criterion, train_loader, epoch,writers, writer, fold)
            #if epoch%3 == 0 or epoch == config['epochs']-1:
            val_dict = validate(model,optimizer,criterion, val_loader, epoch, writers, writer, fold,train_dict['threshold'])
            if val_dict['auroc']>best_auc:
                best_loss = val_dict['loss']
                best_auc = val_dict['auroc']
                
            if early_stopping and ((epoch>20 and best_auc<0.6 and train_dict['auroc'] >0.7) or (epoch>20 and train_dict['auroc'] <0.55)):
                if epoch%3!=0:
                    val_dict = validate(model,optimizer,criterion, val_loader, epoch, writers, writer, fold,train_dict['threshold'])
                    if val_dict['auroc']>best_auc:
                        best_loss = val_dict['loss']
                        best_auc = val_dict['auroc']
                        torch.save(model.state_dict(), os.path.join(f"{logdir}/Fold_{i+1}","best_model.pth"))
                if best_auc<0.55 or  (epoch>20 and train_dict['auroc'] <0.55):
                    break
        fold_val_losses.append(best_loss)
        fold_val_aucs.append(best_auc)
    
    writer.add_scalar('CV_loss', np.mean(fold_val_losses))
    writer.add_scalar('CV_auroc', np.mean(fold_val_aucs))
    writer.add_text('Config', f"Model: {config['model']}, lr: {config['lr']}, batch_size: {config['batch_size']}, nFold: {config['n_fold']}", 0)
    writer.close()
    return {
        "model": model,
        "loss": np.mean(fold_val_losses),
        "auroc" : np.mean(fold_val_aucs),
    }

def train_model_fusion(config, 
                       train_dataset, 
                       model, 
                       checkpoint_dir=None,
                       early_stopping = False, 
                       current_time = datetime.now().strftime("%Y%m%d-%H%M%S"),
                      device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    logdir = f"Logs/{current_time}/{config['run_name']}_LR:{config['lr']}_NF:{config['n_fold']}_E:{config['epochs']}_B:{config['batch_size']}_WD:{config.get('weight_decay', 0)}"  
    # Setup data for K-Folds
    k_folds = config['n_fold']
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=252)
    # TensorBoard
    writers = [SummaryWriter(f"{logdir}/Fold_{i+1}") for i in range(k_folds)]
    writer = SummaryWriter(logdir)
    
    fold_val_losses = []
    fold_val_aucs = []
    original_state_dict = model.state_dict().copy()
    temp_model_path = f"./temp_original_model_f_{config['run_name']}.pth"
    torch.save( original_state_dict, temp_model_path)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(np.arange(len(train_dataset)))):
        
        # Initialize model and optimizer (Reset model to original weights)
        model.load_state_dict(torch.load(temp_model_path, weights_only = True))
            
        model.to(device)    
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0))
        criterion = nn.BCEWithLogitsLoss()
        best_auc = 0
        best_loss = 0
        train_subsampler = Subset(train_dataset, train_ids)
        test_subsampler = Subset(train_dataset, test_ids)

        train_loader = DataLoader(train_subsampler, batch_size=int(config["batch_size"]), shuffle=True)
        val_loader = DataLoader(test_subsampler, batch_size=int(config["batch_size"]), shuffle=False)
        train_dict = {}
        val_dict = {}
        for epoch in range(config['epochs']): 
            train_dict = train_fusion(model, optimizer,criterion, train_loader, epoch,writers, writer, fold, device)
            #if epoch%3 == 0 or epoch == config['epochs']-1:
            val_dict = validate_fusion(model,optimizer,criterion, val_loader, epoch, writers, writer, fold,train_dict['threshold'],device)
            if val_dict['auroc']>best_auc:
                best_loss = val_dict['loss']
                best_auc = val_dict['auroc']
                
            if early_stopping and ((epoch>20 and best_auc<0.6 and train_dict['auroc'] >0.7) or (epoch>20 and train_dict['auroc'] <0.55)):
                if epoch%3!=0:
                    val_dict = validate_fusion(model,optimizer,criterion, val_loader, epoch, writers, writer, fold,train_dict['threshold'], device)
                    if val_dict['auroc']>best_auc:
                        best_loss = val_dict['loss']
                        best_auc = val_dict['auroc']
                        torch.save(model.state_dict(), os.path.join(f"{logdir}/Fold_{i+1}","best_model.pth"))
                if best_auc<0.55 or  (epoch>20 and train_dict['auroc'] <0.55):
                    break
        fold_val_losses.append(best_loss)
        fold_val_aucs.append(best_auc)
    
    try:
        os.remove(temp_model_path)
    except:
        print('removing temp files failed.')
    writer.add_scalar('CV_loss', np.mean(fold_val_losses))
    writer.add_scalar('CV_auroc', np.mean(fold_val_aucs))
    writer.add_text('Config', f"Model: {config['run_name']}, lr: {config['lr']}, batch_size: {config['batch_size']}, nFold: {config['n_fold']}", 0)
    writer.close()
    
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return {
        "model": model,
        "loss": np.mean(fold_val_losses),
        "auroc" : np.mean(fold_val_aucs),
    }
    
#### DATASETS
class ClinicalDataset(Dataset):
    def __init__(self, outcome_df, clinical_data, outcome):
        self.df = outcome_df
        self.outcome = outcome
        self.df = self.df.merge(clinical_data, how = "left", on="case_id")
        lendf = len(self.df)
        self.df = self.df.dropna(subset = [outcome])
        self.df=self.df.fillna(-1)
        self.columns = clinical_data.columns[1:]
        print(f"{lendf - len(self.df)} row deleted due to lack of outcome info. Current length = {len(self.df)}")
    def __len__(self):
        return len(self.df)
    def num_positives(self):
        return self.df[self.outcome].sum(), len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #print(df.columns)
        input_data = row[self.columns].values.astype('float')
        #print(type(input_data))
        input_data = torch.tensor(input_data, dtype=torch.float32)  # Ensuring the correct dtype
        label = torch.tensor(row[self.outcome], dtype=torch.float32)  # Ensure label is a float tensor
        case_id = row['case_id']

        return {"input": input_data, "label": label, "case_id": case_id}
    
class MultimodalDataset(Dataset):
    def __init__(self, df, outcome: str, imaging_df, clinical_data, transform=None, image_folder = 'img', add_post = True, root = "/msklab/Data/ResampledFinalSarcomaDataV5/cropped", include_image = True):
        self.df = df
        self.transform = transform
        self.outcome = outcome
        self.clinical_data = clinical_data
        self.clinical_data = self.clinical_data.fillna(-1)
        #self.clinical_data = self.clinical_data.dropna(subset = [outcome])
        self.df = df.merge(imaging_df, how = "left", on="case_id")
        self.image_folder = image_folder
        self.add_post = add_post
        self.columns = clinical_data.columns[1:]
        self.root = root
        self.include_image = include_image
        lendf = len(self.df)
        self.df = self.df.dropna(
            #how='all',  # Specify 'all' to drop rows where all specified columns are NaN
            how='any',
            subset=['t2_series_description'],
            #subset=['t2_series_description', 't1ac_series_description', 't1bc_series_description', 't1na_series_description']
        )
        self.df = self.df.dropna(subset = [outcome])
        print(f"{lendf - len(self.df)} row deleted due to lack of images. Current length = {len(self.df)}")
    def __len__(self):
        return len(self.df)
    def num_positives(self):
        return self.df[self.outcome].sum(), len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images_dict = {
            "t2": "None",
            "t1ac": "None",
            "t1bc" : "None",
            "t1na" : "None",
        }
        if self.add_post:
            images_dict = {
                "t2": "None",
                "t1ac": "None",
                "t1bc" : "None",
                "t1na" : "None",
                "pt2": "None",
                "pt1ac": "None",
                "pt1bc" : "None",
                "pt1na" : "None",
            }
            if not pd.isna(row['post_t2_series_description']):
                images_dict['pt2'] = str(row['case_id'])+"_"+row['post_t2_series_description']+".npy"
            if not pd.isna(row['post_t1ac_series_description']):
                images_dict['pt1ac'] = str(row['case_id'])+"_"+row['post_t1ac_series_description']+".npy"
            if not pd.isna(row['post_t1bc_series_description']):
                images_dict['pt1bc'] = str(row['case_id'])+"_"+row['post_t1bc_series_description']+".npy"
            if not pd.isna(row['post_t1na_series_description']):
                images_dict['pt1na'] = str(row['case_id'])+"_"+row['post_t1na_series_description']+".npy"
        if not pd.isna(row['t2_series_description']):
            images_dict['t2'] = str(row['case_id'])+"_"+row['t2_series_description']+".npy"
        if not pd.isna(row['t1ac_series_description']):
            images_dict['t1ac'] = str(row['case_id'])+"_"+row['t1ac_series_description']+".npy"
        if not pd.isna(row['t1bc_series_description']):
            images_dict['t1bc'] = str(row['case_id'])+"_"+row['t1bc_series_description']+".npy"
        if not pd.isna(row['t1na_series_description']):
            images_dict['t1na'] = str(row['case_id'])+"_"+row['t1na_series_description']+".npy"
        if self.include_image:
            for img, path in images_dict.items():   
                if path!="None": 
                    images_dict[img] = os.path.join(self.root , self.image_folder, path)

            name = row['case_id']
            blank_img = np.zeros([32,128,128])
            for img, path in images_dict.items():
                if not os.path.exists(path):
                    images_dict[img] = np.array([None])
                else:
                    images_dict[img] = np.load(images_dict[img])
                    blank_img = np.zeros(images_dict[img].shape)
                #### Add logic to deal with missing sequences
            for img, path in images_dict.items():
                if images_dict[img].any() == None:
                    images_dict[img] = blank_img   

            min_depth = min(img.shape[0] for img in images_dict.values())
            min_height = min(img.shape[1] for img in images_dict.values())
            min_width = min(img.shape[2] for img in images_dict.values())

            # Crop each image to the center with the minimum dimensions and stack them
            image = np.stack([
                img[
                    (img.shape[0] - min_depth) // 2 : (img.shape[0] + min_depth) // 2,
                    (img.shape[1] - min_height) // 2 : (img.shape[1] + min_height) // 2,
                    (img.shape[2] - min_width) // 2 : (img.shape[2] + min_width) // 2
                ]
                for img in images_dict.values()
            ])

            D = image.shape[1]  # Depth of the 3D image
            slices_idx = np.linspace(0, D - 1, 16, dtype=int)

            image = image[:,slices_idx,:,:]
            label = row[self.outcome]  # Assuming binary labels are in the 'label' column
            if self.transform:
                image = self.transform(image)
            
        else:
            image = torch.zeros([8, 16,128,128])
            label = row[self.outcome]  # Assuming binary labels are in the 'label' column
            if self.transform:
                image = self.transform(image)
           
        pt_id = row['case_id']
        input_data = self.clinical_data.loc[self.clinical_data.case_id == pt_id][self.columns].values.astype('float')
            
        return {"img": image.astype(torch.float32), 
                "input" : torch.tensor(input_data, dtype=torch.float).squeeze(),
                "label":torch.tensor(label, dtype=torch.float),
               "case_id": row['case_id']}


class EightChannelNpyDataset(Dataset):
    """
    Dataset for a single 8-channel .npy image per patient.

    Expected:
      - One file per patient in `root/image_folder/` (or directly under root if image_folder="").
      - Default filename pattern: {case_id}.npy
      - The loaded array must be shaped (8, D, H, W) or (D, H, W, 8).
        If (D, H, W, 8), it will be transposed to (8, D, H, W).

    Output matches the notebooks / other datasets:
      {"img": Tensor[8, 16, 128, 128] (after transforms),
       "input": Tensor[num_features],
       "label": Tensor[float],
       "case_id": case_id}
    """
    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str,
        clinical_data: pd.DataFrame,
        transform=None,
        root: str = "",
        image_folder: str = "",
        filename_template: str = "{case_id}.npy",
        include_image: bool = True,
    ):
        self.df = df.copy()
        self.transform = transform
        self.outcome = outcome
        self.clinical_data = clinical_data.copy()
        self.clinical_data = self.clinical_data.fillna(-1)
        self.columns = self.clinical_data.columns[1:]
        self.root = root
        self.image_folder = image_folder
        self.filename_template = filename_template
        self.include_image = include_image

        # Drop rows missing labels (consistent with other datasets)
        self.df = self.df.dropna(subset=[outcome])

    def __len__(self):
        return len(self.df)

    def num_positives(self):
        return self.df[self.outcome].sum(), len(self.df)

    def _resolve_path(self, case_id):
        rel = self.filename_template.format(case_id=case_id)
        if self.image_folder:
            return os.path.join(self.root, self.image_folder, rel)
        return os.path.join(self.root, rel)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        label = row[self.outcome]

        if self.include_image:
            path = self._resolve_path(case_id)
            if not os.path.exists(path):
                # Match prior behavior: create a blank tensor if missing
                image = np.zeros([8, 16, 128, 128], dtype=np.float32)
            else:
                image = np.load(path)

                # Accept (D, H, W, 8) or (8, D, H, W)
                if image.ndim == 4 and image.shape[-1] == 8 and image.shape[0] != 8:
                    image = np.transpose(image, (3, 0, 1, 2))

                # Ensure channels-first
                if image.ndim != 4 or image.shape[0] != 8:
                    raise ValueError(
                        f"Expected 8-channel npy for case_id={case_id} with shape (8,D,H,W) "
                        f"or (D,H,W,8). Got shape {getattr(image, 'shape', None)} at: {path}"
                    )

                # Match notebook behavior: pick 16 evenly spaced slices along depth
                D = image.shape[1]
                slices_idx = np.linspace(0, D - 1, 16, dtype=int)
                image = image[:, slices_idx, :, :]

                if self.transform:
                    image = self.transform(image)
        else:
            image = torch.zeros([8, 16, 128, 128])
            if self.transform:
                image = self.transform(image)

        input_data = (
            self.clinical_data.loc[self.clinical_data.case_id == case_id][self.columns]
            .values.astype("float")
        )

        return {
            "img": image.astype(torch.float32),
            "input": torch.tensor(input_data, dtype=torch.float).squeeze(),
            "label": torch.tensor(label, dtype=torch.float),
            "case_id": case_id,
        }


class MultimodalWithAnySeqDataset(Dataset):
    def __init__(self, df, outcome: str, imaging_df, clinical_data, transform=None, image_folder = 'img', add_post = True, root = "/msklab/Data/ResampledFinalSarcomaDataV5/cropped"):
        self.df = df
        self.transform = transform
        self.outcome = outcome
        self.clinical_data = clinical_data
        self.clinical_data = self.clinical_data.fillna(-1)
        #self.clinical_data = self.clinical_data.dropna(subset = [outcome])
        self.df = df.merge(imaging_df, how = "left", on="case_id")
        self.image_folder = image_folder
        self.add_post = add_post
        self.columns = clinical_data.columns[1:]
        self.root = root
        lendf = len(self.df)
        self.df = self.df.dropna(
            how='all',  # Specify 'all' to drop rows where all specified columns are NaN
            #how='any',
            #subset=['t2_series_description'],
            subset=['t2_series_description', 't1ac_series_description', 't1bc_series_description', 't1na_series_description']
        )
        self.df = self.df.dropna(subset = [outcome])
        print(f"{lendf - len(self.df)} row deleted due to lack of images. Current length = {len(self.df)}")
    def __len__(self):
        return len(self.df)
    def num_positives(self):
        return self.df[self.outcome].sum(), len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images_dict = {
            "t2": "None",
            "t1ac": "None",
            "t1bc" : "None",
            "t1na" : "None",
        }
        if self.add_post:
            images_dict = {
                "t2": "None",
                "t1ac": "None",
                "t1bc" : "None",
                "t1na" : "None",
                "pt2": "None",
                "pt1ac": "None",
                "pt1bc" : "None",
                "pt1na" : "None",
            }
            if not pd.isna(row['post_t2_series_description']):
                images_dict['pt2'] = str(row['case_id'])+"_"+row['post_t2_series_description']+".npy"
            if not pd.isna(row['post_t1ac_series_description']):
                images_dict['pt1ac'] = str(row['case_id'])+"_"+row['post_t1ac_series_description']+".npy"
            if not pd.isna(row['post_t1bc_series_description']):
                images_dict['pt1bc'] = str(row['case_id'])+"_"+row['post_t1bc_series_description']+".npy"
            if not pd.isna(row['post_t1na_series_description']):
                images_dict['pt1na'] = str(row['case_id'])+"_"+row['post_t1na_series_description']+".npy"
        if not pd.isna(row['t2_series_description']):
            images_dict['t2'] = str(row['case_id'])+"_"+row['t2_series_description']+".npy"
        if not pd.isna(row['t1ac_series_description']):
            images_dict['t1ac'] = str(row['case_id'])+"_"+row['t1ac_series_description']+".npy"
        if not pd.isna(row['t1bc_series_description']):
            images_dict['t1bc'] = str(row['case_id'])+"_"+row['t1bc_series_description']+".npy"
        if not pd.isna(row['t1na_series_description']):
            images_dict['t1na'] = str(row['case_id'])+"_"+row['t1na_series_description']+".npy"

        for img, path in images_dict.items():   
            if path!="None": 
                images_dict[img] = os.path.join(self.root , self.image_folder, path)
        
        name = row['case_id']
        blank_img = np.zeros([32,128,128])
        for img, path in images_dict.items():
            if not os.path.exists(path):
                images_dict[img] = np.array([None])
            else:
                images_dict[img] = np.load(images_dict[img])
                blank_img = np.zeros(images_dict[img].shape)
            #### Add logic to deal with missing sequences
        for img, path in images_dict.items():
            if images_dict[img].any() == None:
                images_dict[img] = blank_img   
                
        min_depth = min(img.shape[0] for img in images_dict.values())
        min_height = min(img.shape[1] for img in images_dict.values())
        min_width = min(img.shape[2] for img in images_dict.values())

        # Crop each image to the center with the minimum dimensions and stack them
        image = np.stack([
            img[
                (img.shape[0] - min_depth) // 2 : (img.shape[0] + min_depth) // 2,
                (img.shape[1] - min_height) // 2 : (img.shape[1] + min_height) // 2,
                (img.shape[2] - min_width) // 2 : (img.shape[2] + min_width) // 2
            ]
            for img in images_dict.values()
        ])
        
        D = image.shape[1]  # Depth of the 3D image
        slices_idx = np.linspace(0, D - 1, 16, dtype=int)
        
        image = image[:,slices_idx,:,:]
        label = row[self.outcome]  # Assuming binary labels are in the 'label' column
        if self.transform:
            image = self.transform(image)
            
        pt_id = row['case_id']
        input_data = self.clinical_data.loc[self.clinical_data.case_id == pt_id][self.columns].values.astype('float')
            
        return {"img": image.astype(torch.float32), 
                "input" : torch.tensor(input_data, dtype=torch.float).squeeze(),
                "label":torch.tensor(label, dtype=torch.float),
               "case_id": row['case_id']}

        
    
#### Model architectures
class ClinicalClassifier(nn.Module):
    def __init__(self, input_size):
        super(ClinicalClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        #x = self.sigmoid(self.output(x))
        return x

    
class SimpleClinicalClassifier(nn.Module):
    def __init__(self, input_size, fc_size = 32):
        super(SimpleClinicalClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, fc_size)
        self.output = nn.Linear(fc_size, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.output(x)
        return x

    
class IntermediateFusionModel(nn.Module):
    def __init__(self, base_model, input_size, clinical_weights_path = None, imaging_weights_path = None):
        super(IntermediateFusionModel, self).__init__()
        # Load and modify ResNetfor feature extraction
        self.resnet = base_model
        self.resnet.fc = nn.Identity()  # Remove the classification layer
        # Clinical classifier for tabular data
        self.clinical_net = ClinicalClassifier(input_size)
        if clinical_weights_path != None:
            clinical_weights = torch.load(clinical_weights_path, weights_only = True)
            clinical_weights = {k.replace("clinical_net.", ""): v for k, v in clinical_weights.items()}
            self.clinical_net.load_state_dict(clinical_weights, strict = False)
            print("clinical weights successfully loaded.")
        if imaging_weights_path != None:
            imaging_weights = torch.load(imaging_weights_path, weights_only = True)
            self.resnet.load_state_dict(imaging_weights, strict = False)
            print("imaging weights successfully loaded.")
        self.clinical_net.output = nn.Identity() 
        with torch.no_grad():
            dummy_image = torch.randn(1, 8, 16, 224, 224).to(next(self.resnet.parameters()).device)
            base_model_output_size = self.resnet(dummy_image).shape[1]
            
        
        # Fusion layer
        self.fusion_layer = nn.Linear(base_model_output_size + 16, 128)  # ResNet + ClinicalClassifier (16)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, image, clinical_data):
        # Extract features from ResNet
        image_features = self.resnet(image)  
        # Extract features from ClinicalClassifier
        clinical_features = self.clinical_net(clinical_data)
        #print(image_features.shape, clinical_features.shape)
        # Concatenate features
        combined_features = torch.cat((image_features, clinical_features), dim=1)
        # Fusion and output
        x = torch.relu(self.fusion_layer(combined_features))
        x = self.output_layer(x)
        return x
    
class SimpleIntermediateFusionModel(nn.Module):
    def __init__(self, base_model, input_size, clinical_weights_path = None, imaging_weights_path = None, clin_fc_size = 32):
        super(SimpleIntermediateFusionModel, self).__init__()
        # Load and modify ResNetfor feature extraction
        self.resnet = base_model
        self.resnet.fc = nn.Identity()  # Remove the classification layer
        # Clinical classifier for tabular data
        self.clinical_net = SimpleClinicalClassifier(input_size, clin_fc_size)
        if clinical_weights_path != None:
            clinical_weights = torch.load(clinical_weights_path, weights_only = True)
            clinical_weights = {k.replace("clinical_net.", ""): v for k, v in clinical_weights.items()}
            self.clinical_net.load_state_dict(clinical_weights, strict = False)
            print("clinical weights successfully loaded.")
        if imaging_weights_path != None:
            imaging_weights = torch.load(imaging_weights_path, weights_only = True)
            self.resnet.load_state_dict(imaging_weights, strict = False)
            print("imaging weights successfully loaded.")
        self.clinical_net.output = nn.Identity() 
        with torch.no_grad():
            dummy_image = torch.randn(1, 8, 16, 224, 224).to(next(self.resnet.parameters()).device)
            base_model_output_size = self.resnet(dummy_image).shape[1]
            
        
        # Fusion layer
        #self.fusion_layer = nn.Linear(base_model_output_size + 16, 128)  # ResNet + ClinicalClassifier (16)
        self.output_layer = nn.Linear(base_model_output_size + clin_fc_size, 1)

    def forward(self, image, clinical_data):
        # Extract features from ResNet
        image_features = self.resnet(image)  
        # Extract features from ClinicalClassifier
        clinical_features = self.clinical_net(clinical_data)
        #print(image_features.shape, clinical_features.shape)
        # Concatenate features
        combined_features = torch.cat((image_features, clinical_features), dim=1)
        # Fusion and output
        #x = torch.relu(self.fusion_layer(combined_features))
        x = self.output_layer(combined_features)
        return x

class EarlyFusionModel(nn.Module):
    def __init__(self, base_model, input_size, imaging_weights_path = None):
        super(EarlyFusionModel, self).__init__()
        # Load and modify ResNetfor feature extraction
        self.resnet = base_model
        self.resnet.fc = nn.Identity()  # Remove the classification layer
        
        if imaging_weights_path != None:
            imaging_weights = torch.load(imaging_weights_path, weights_only = True)
            self.resnet.load_state_dict(imaging_weights, strict = False)
            print("imaging weights successfully loaded.")
        
        with torch.no_grad():
            dummy_image = torch.randn(1, 8, 16, 224, 224).to(next(self.resnet.parameters()).device)
            base_model_output_size = self.resnet(dummy_image).shape[1]
            
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Fusion layer
        self.fusion_layer = nn.Linear(base_model_output_size + input_size, 128)  # ResNet + ClinicalClassifier (16)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, image, clinical_data):
        # Extract features from ResNet
        image_features = self.resnet(image)  
        # Extract features from ClinicalClassifier
        #print(image_features.shape, clinical_features.shape)
        # Concatenate features
        combined_features = torch.cat((image_features, clinical_data), dim=1)
        # Fusion and output
        x = torch.relu(self.fusion_layer(combined_features))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x
    
class SimpleEarlyFusionModel(nn.Module):
    def __init__(self, base_model, input_size, imaging_weights_path = None):
        super(SimpleEarlyFusionModel, self).__init__()
        # Load and modify ResNetfor feature extraction
        self.resnet = base_model
        self.resnet.fc = nn.Identity()  # Remove the classification layer
        
        if imaging_weights_path != None:
            imaging_weights = torch.load(imaging_weights_path, weights_only = True)
            self.resnet.load_state_dict(imaging_weights, strict = False)
            print("imaging weights successfully loaded.")
        
        with torch.no_grad():
            dummy_image = torch.randn(1, 8, 16, 224, 224).to(next(self.resnet.parameters()).device)
            base_model_output_size = self.resnet(dummy_image).shape[1]
            
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Fusion layer
        self.fusion_layer = nn.Linear(base_model_output_size + input_size, 128)  # ResNet + ClinicalClassifier (16)
        self.layer1 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, image, clinical_data):
        # Extract features from ResNet
        image_features = self.resnet(image)  
        # Extract features from ClinicalClassifier
        #print(image_features.shape, clinical_features.shape)
        # Concatenate features
        combined_features = torch.cat((image_features, clinical_data), dim=1)
        # Fusion and output
        x = torch.relu(self.fusion_layer(combined_features))
        x = torch.relu(self.layer1(x))
        x = self.output(x)
        return x
    
class LateFusionModel(nn.Module):
    def __init__(self, imaging_model, clinical_model, imaging_weights_path = None, clinical_weights_path = None):
        super(LateFusionModel, self).__init__()
        # Load and modify ResNetfor feature extraction
        self.resnet = imaging_model
        
        # Clinical classifier for tabular data
        self.clinical_net = clinical_model
        if clinical_weights_path != None:
            clinical_weights = torch.load(clinical_weights_path, weights_only = True)
            clinical_weights = {k.replace("clinical_net.", ""): v for k, v in clinical_weights.items()}
            self.clinical_net.load_state_dict(clinical_weights, strict = True)
            print("clinical weights successfully loaded.")
        if imaging_weights_path != None:
            imaging_weights = torch.load(imaging_weights_path, weights_only = True)
            self.resnet.load_state_dict(imaging_weights, strict = True)
            print("imaging weights successfully loaded.")
            
        for param in self.clinical_net.parameters():
            param.requires_grad = False
        
        for param in self.resnet.parameters():
            param.requires_grad = False 
        # Fusion layer
        self.fusion_layer = nn.Linear(2, 16)  # ResNet + ClinicalClassifier (16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, image, clinical_data):
        # Extract features from ResNet
        image_features = self.resnet(image)  
        # Extract features from ClinicalClassifier
        clinical_features = self.clinical_net(clinical_data)
        #print(image_features.shape, clinical_features.shape)
        # Concatenate features
        combined_features = torch.cat((image_features, clinical_features), dim=1)
        # Fusion and output
        x = torch.relu(self.fusion_layer(combined_features))
        x = self.output_layer(x)
        return x
    


    
    
