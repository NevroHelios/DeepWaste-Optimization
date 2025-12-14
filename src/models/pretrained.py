
import torchvision
import torch.nn as nn

def get_pretrained_model(num_classes: int, freeze_strategy: str, freeze_layers: int = 0):
    # mobilenet cause its fast
    model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
    
    if freeze_strategy == 'freeze_all_except_last':
        for param in model.parameters():
            param.requires_grad = False
    elif freeze_strategy == 'freeze_k_layers':
        # MobileNetV3 features => 13 children + fc_first
        features = list(model.features.children())
        if freeze_layers > len(features):
            freeze_layers = len(features)
        
        for i in range(freeze_layers):
            for param in features[i].parameters():
                param.requires_grad = False
                
    elif freeze_strategy == 'finetune_all':
        for param in model.parameters():
            param.requires_grad = True
    
    last_layer_idx = 3
    in_features = model.classifier[last_layer_idx].in_features
    model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)
    
    if freeze_strategy == 'freeze_all_except_last':
        for param in model.classifier[last_layer_idx].parameters():
            param.requires_grad = True

    return model
