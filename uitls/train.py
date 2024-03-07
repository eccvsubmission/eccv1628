import gc
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score
from torchmetrics.functional import average_precision

def train_loop(model, train_loader,
               criterion, optimizer,
               accelerator, scheduler=None,
               verbose=False,device_nr:int=0):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    
    device = f"cuda:{device_nr}"
    model = model.to(device)
    for i, x in (enumerate(train_loader)):
        with accelerator.accumulate(model):
            x = {k:v.to(device) for (k,v) in x.items()}
            targets = x["label"]
            outputs = model(x)
            print(outputs.shape)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
            # Compute accuracy and accumulate loss per batch
            total_loss += loss.item() 
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == torch.argmax(targets, dim=1)).sum().item() 
            y_pred.extend(predicted.cpu().tolist()), y_true.extend(torch.argmax(targets, dim=1).cpu().tolist())

    # Compute epoch accuracy and loss
    accuracy = correct_predictions / total_predictions
    epoch_loss = total_loss / (i+1)
    gc.collect(), torch.cuda.empty_cache()
    if verbose:
        print(f"Train Accuracy: {accuracy:.4f}")
        print(f"Train Loss: {epoch_loss:.4f}") 
    
    num_classes = train_loader.dataset.num_classes
    y_pred_oh = torch.nn.functional.one_hot(torch.tensor(y_pred), num_classes)
    y_true_oh = torch.nn.functional.one_hot(torch.tensor(y_true), num_classes)
    
    result_dict = {"epoch_loss": epoch_loss,
                   "accuracy": accuracy,
                   "precision" : precision_score(y_true, y_pred, average="macro"),
                  "f1": f1_score(y_true, y_pred, average="macro"),
                  "average_precision_mean": average_precision(y_pred_oh, y_true_oh, num_classes=num_classes),
                   "average_precision": average_precision(y_pred_oh, y_true_oh, num_classes=num_classes,average=None)}
    return model, optimizer, scheduler, result_dict
