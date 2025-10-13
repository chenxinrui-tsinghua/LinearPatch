from rotate_utils.hadamard_utils import get_hadamard_matrix
import torch
from tqdm import tqdm


def get_scale_params(model, trainloader, i, j, rotate, device):
    model = model.to(device)
    num_layers = len(model.model.layers)
    d_clip = [torch.zeros(1).to(device) for _ in range(num_layers)]
    scale_params = torch.zeros(1).to(device)
    def hook(module, input, output, layer_name, rotate):
        if rotate:
            d_clip[layer_name] = torch.matmul(input[0], get_hadamard_matrix(input[0].shape[-1], input[0].device).half())
        else:
            d_clip[layer_name] = input[0]
    handles = []

    for l, layer in enumerate(model.model.layers):
        if l==i or l==j:
            handle = layer.register_forward_hook(lambda module, input, output, layer_name=l, rotate=rotate: hook(module, input, output, layer_name, rotate))
            handles.append(handle)


    num_samples = num_sample = 128
    calibration_loop = tqdm(enumerate(trainloader, start=0), desc="Calibrating", total=num_samples)
    for num, batch in calibration_loop:
        batch = batch[0].to(device)
        try:
            with torch.no_grad():
                output = model(batch)
        except IndexError:
            pass
        scale_param = (d_clip[j].abs().mean(dim=0, keepdim=True).mean(dim=1, keepdim=True) /
            d_clip[i].abs().mean(dim=0, keepdim=True).mean(dim=1, keepdim=True))  # [1,1,4096]
        scale_params = scale_params + scale_param

        loss_before = torch.abs(scale_param-1).mean()
        loss_after = torch.abs(scale_param/(scale_params/num)-1).mean()

        calibration_loop.set_postfix({"Gap before": loss_before.detach().item(), "Gap after": loss_after.detach().item()})

        num_sample -= 1
        if not num_sample:
            break

    scale_params = scale_params / num_samples

    for handle in handles:
        handle.remove()

    torch.cuda.empty_cache()
    return scale_params[0][0]

def get_pruned_layer_with_cosine(model, trainloader, num_to_prune, device):
    model = model.to(device)
    num_layers = len(model.model.layers)
    max_start = num_layers - num_to_prune
    act = [torch.zeros(1).to(device) for _ in range(num_layers)]
    cosine_sim = [torch.zeros(1).to(device) for _ in range(max_start)]

    def hook(module, input, output, layer_name):
        act[layer_name] = input[0]
    handles = []

    for l, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(lambda module, input, output, layer_name=l: hook(module, input, output, layer_name))
        handles.append(handle)

    num_samples = num_sample = 128
    select_loop = tqdm(enumerate(trainloader, start=0), desc="Selecting", total=num_samples)
    for num, batch in select_loop:
        batch = batch[0].to(device)
        try:
            with torch.no_grad():
                output = model(batch)
        except IndexError:
            pass
        num_sample -= 1

        for i in range(1, max_start):
            cosine_sim[i] += torch.cosine_similarity(act[i], act[i+num_to_prune]).mean()

        if not num_sample:
            break

    for handle in handles:
        handle.remove()

    cosine_sim = [i.item()/num_samples for i in cosine_sim]
    max_sim = max(cosine_sim)
    start_l = cosine_sim.index(max_sim)
    end_l = start_l + num_to_prune

    torch.cuda.empty_cache()
    return num_to_prune, start_l, end_l, max_sim



def select_layer(model, trainloader, num_to_prune, insert_type, dev):
    import time
    tick = time.time()
    print("Start Calibration")

    layer, start_l, end_l, max_value = get_pruned_layer_with_cosine(model, trainloader, num_to_prune, dev)
    print(f"Prune layer in  #Layer[{start_l}, {end_l})")

    layer_select_time_cost = time.time() - tick
    tick = time.time()

    if insert_type != 'none':
        scale_params = get_scale_params(model, trainloader, start_l, end_l, insert_type == "rotate", dev)
    else:
        scale_params = torch.ones(4096, dtype=torch.float32).to(dev)

    scale_calculate_time_cost = time.time() - tick

    return layer, start_l, end_l, max_value, scale_params, layer_select_time_cost, scale_calculate_time_cost