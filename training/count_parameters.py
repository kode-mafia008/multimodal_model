from models import MultiModalSentimentModel

def count_parameters(model):
    params_dict = {
        'text_encoder':0,
        'video_encoder':0,
        'audio_encoder':0,
        'fusion_layer':0,
        'emotion_classifier':0,
        'sentiment_classifier':0
    }
    
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Count the number of parameters in the tensor
            param_count = param.numel()
            total_params += param.numel()

        if 'text_encoder' in name:
            params_dict['text_encoder'] += param_count
        elif 'video_encoder' in name:
            params_dict['video_encoder'] += param_count
        elif 'audio_encoder' in name:
            params_dict['audio_encoder'] += param_count
        elif 'fusion_layer' in name:
            params_dict['fusion_layer'] += param_count 
        elif 'emotion_classifier' in name:
            params_dict['emotion_classifier'] += param_count
        elif 'sentiment_classifier' in name:
            params_dict['sentiment_classifier'] += param_count
    
    return total_params,params_dict

if __name__ == "__main__":
    model = MultiModalSentimentModel()
    params_dicts,total_params = count_parameters(model)

    # Print the total number of parameters
    print(f"Total number of parameters: {total_params}")
    
    # Print the number of parameters for each component
    for component, count in params_dicts.items():
        print(f"{component:20s}: {count:,} parameters")
    
    # Print total no of trainable parameters
    print(f"Total number of trainable parameters: {total_params:,}")
    
    