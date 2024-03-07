import torch
from torch import nn, Tensor
from typing import Dict, List, Optional
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

class OneStreamVMAE(nn.Module):
    def __init__(self,
                 label2id,
                 id2label,
                 model_ckpt =  "MCG-NJU/videomae-base-finetuned-kinetics",
                 stream_name = "video"):
        super().__init__()
        self.vmae = VideoMAEForVideoClassification.from_pretrained(
                            model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True,)
        self.stream = stream_name
    def forward(self, x):
        x = self.vmae(x[self.stream]).logits#.permute(0,2,1,3,4)).logits
        return x
    
def freeze_params(model):
    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Print to verify that all parameters are frozen
    for name, param in model.named_parameters():
        if "attention" in name:
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad =True
        if "attn" in name:
            param.requires_grad = True
        if "head" in name:
            param.requires_grad =True
            
