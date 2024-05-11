import torch

checkpoint = torch.load('outputs/fourierkan-blender-lego/xkanerf/2024-05-11_092040/nerfstudio_models/step-000029999.ckpt')

def count_parameters(obj, prefix=''):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            params = count_parameters(value, new_prefix)
            total_params += params['total']
            trainable_params += params['trainable']
            non_trainable_params += params['non_trainable']
    elif isinstance(obj, torch.Tensor):
        num_params = obj.numel()
        if obj.requires_grad:
            if 'layers' in prefix:
                print(f"{prefix} (trainable): {num_params}")
                trainable_params += num_params
        else:
            if 'layers' in prefix:
                print(f"{prefix} (non-trainable): {num_params}")
                non_trainable_params += num_params
        total_params += num_params
    # else:
    #     print(f"{prefix}: {type(obj)}")

    return {'total': total_params, 'trainable': trainable_params, 'non_trainable': non_trainable_params}

# # 统计所有参数
result = count_parameters(checkpoint)

print(f"Total parameters: {result['total']}")
print(f"Trainable parameters: {result['trainable']}")
print(f"Non-trainable parameters: {result['non_trainable']}")