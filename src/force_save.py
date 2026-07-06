import torch
import torch.nn as nn
import os

# This re-creates the structure so we can save the model
NUM_CUSTOM_CLASSES = 5
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=False)
in_features = model.blocks[5].proj.in_features
model.blocks[5].proj = nn.Linear(in_features, NUM_CUSTOM_CLASSES)

# This assumes the 'model' object is still defined in your memory
# If the terminal closes, the weights are gone. 
# BUT, if you haven't closed the terminal, the weights might still be in the process!
# Actually, the safest way right now: