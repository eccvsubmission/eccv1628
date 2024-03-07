import gc
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score
from torchmetrics.functional import average_precision


def eval_loop(model,
              val_loader,
        criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.0),
        verbose=False, device_nr:int=0):
    
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    device = f"cuda:{device_nr}"
    model = model.to(device)
    for i, x in (enumerate(val_loader)):
        x = {k:v.to(device) for (k,v) in x.items()}
        targets = x["label"]
        outputs  = model(x)
        loss = criterion(outputs, targets)
        total_loss += loss.item() 
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += targets.size(0)
        correct_predictions += (predicted == torch.argmax(targets, dim=1)).sum().item() 
        y_pred.extend(predicted.cpu().tolist()), y_true.extend(torch.argmax(targets, dim=1).cpu().tolist())

    gc.collect(), torch.cuda.empty_cache()
    
    num_classes = val_loader.dataset.num_classes
    y_pred_oh = torch.nn.functional.one_hot(torch.tensor(y_pred), num_classes)
    y_true_oh = torch.nn.functional.one_hot(torch.tensor(y_true), num_classes)
    
    # Compute epoch accuracy and loss
    accuracy = correct_predictions / total_predictions
    epoch_loss = total_loss / (i+1)
    result_dict = {"loss": epoch_loss,
                   "accuracy": accuracy,
                   "precision": precision_score(y_true, y_pred, average="macro"),
                   "f1": f1_score(y_true, y_pred, average="macro"),
                   "mAP": average_precision(y_pred_oh, y_true_oh, num_classes=num_classes).item(),
                   "average_precision": {val_loader.dataset.id2label[k]:v for (k,v) in enumerate([x.item() for x in average_precision(y_pred_oh, y_true_oh, num_classes=num_classes,average=None)])}}
    return result_dict
