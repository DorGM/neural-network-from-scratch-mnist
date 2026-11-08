"""Multilayer network for MNIST with explicit forward and backward passes.

Pixels are divided by 255 for every run. Hidden layers are affine + ReLU.
Batch norm, when enabled, sits after ReLU, reduces over the batch, and uses
running statistics at evaluation. The output layer is affine + softmax.
The loss is mean cross-entropy, plus an L2 term on the weights when lambd > 0.
Parameter updates are plain gradient steps. PyTorch supplies tensors only.
Weight initialization is the explicit argument legacy_fan_in or he. It is not
chosen from use_bnorm.
"""

import math
from pathlib import Path

import torch
from torchvision import datasets

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_LAYER_DIMS = [784, 20, 7, 5, 10]
# Every variant uses this scaling. Raw 0..255 pixels are not a second convention.
PIXEL_SCALE = 255.0
# legacy_fan_in is randn / fan_in. he is randn * sqrt(2 / fan_in).
# Neither scheme is selected from use_bnorm.
INITIALIZATIONS = ("legacy_fan_in", "he")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_train_val_indices(n_examples, val_fraction=0.2, seed=21):
    """Seeded column split of the official training set. The test set is not used."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(n_examples, generator=generator)
    n_val = int(val_fraction * n_examples)
    if n_val < 1 or n_val >= n_examples:
        raise ValueError("Validation split is empty or leaves no training rows.")
    train_index = permutation[n_val:]
    val_index = permutation[:n_val]
    return train_index, val_index


def load_mnist(root=None, val_fraction=0.2, seed=21):
    """Load official MNIST train and test splits and hold out validation from train.

    Download happens here, through torchvision, and only when this function is called.
    Images are float tensors in [0, 1], shaped (features, examples). Labels are one-hot
    with shape (10, examples).
    """
    root = PROJECT_DIR / "data" if root is None else Path(root)
    train_set = datasets.MNIST(root=str(root), train=True, download=True)
    test_set = datasets.MNIST(root=str(root), train=False, download=True)

    x_all = train_set.data.float().view(train_set.data.shape[0], -1).T / PIXEL_SCALE
    y_all_idx = train_set.targets.long()
    x_test = test_set.data.float().view(test_set.data.shape[0], -1).T / PIXEL_SCALE
    y_test_idx = test_set.targets.long()

    train_index, val_index = split_train_val_indices(x_all.shape[1], val_fraction, seed)
    y_all = torch.eye(10)[y_all_idx].T.contiguous()
    y_test = torch.eye(10)[y_test_idx].T.contiguous()
    return {
        "x_train": x_all[:, train_index].contiguous(),
        "y_train": y_all[:, train_index].contiguous(),
        "x_val": x_all[:, val_index].contiguous(),
        "y_val": y_all[:, val_index].contiguous(),
        "x_test": x_test.contiguous(),
        "y_test": y_test.contiguous(),
        "train_index": train_index,
        "val_index": val_index,
        "normalization": "pixel / 255",
    }


def _layer_count(parameters):
    return sum(name.startswith("W") for name in parameters)


def initialize_parameters(layer_dims, use_bnorm=False, device=None, initialization="legacy_fan_in", generator=None):
    """Draw weights with an explicit scheme. use_bnorm does not choose the scheme.

    The historical runs used He scaling only when batch norm was on. Reproducing
    that confounded setup means the caller passes he for those variants and
    legacy_fan_in for the others. Gamma and beta are ones and zeros, so they do
    not consume the weight generator. Running statistics start at mean 0 and
    variance 1 and are not gradient parameters.
    """
    if initialization not in INITIALIZATIONS:
        raise ValueError(f"initialization must be one of {INITIALIZATIONS}.")
    device = get_device() if device is None else torch.device(device)
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("The initialization generator must be a CPU generator.")
    parameters = {}
    last_hidden = len(layer_dims) - 2
    for index in range(len(layer_dims) - 1):
        fan_in = layer_dims[index]
        fan_out = layer_dims[index + 1]
        if initialization == "he":
            scale = math.sqrt(2.0 / fan_in)
        else:
            scale = 1.0 / fan_in
        number = index + 1
        # CPU draws keep one stream for every device. Gamma and beta are not drawn.
        weights = torch.randn(fan_out, fan_in, generator=generator) * scale
        parameters[f"W{number}"] = weights.to(device)
        parameters[f"b{number}"] = torch.zeros(fan_out, 1, device=device)
        if use_bnorm and index < last_hidden:
            parameters[f"gamma{number}"] = torch.ones(fan_out, 1, device=device)
            parameters[f"beta{number}"] = torch.zeros(fan_out, 1, device=device)
            parameters[f"running_mean{number}"] = torch.zeros(fan_out, 1, device=device)
            parameters[f"running_var{number}"] = torch.ones(fan_out, 1, device=device)
    return parameters


def seeded_parameters(layer_dims, use_bnorm, initialization, seed, device):
    """Same seed and scheme produce the same W and b with or without batch norm."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return initialize_parameters(
        layer_dims,
        use_bnorm=use_bnorm,
        device=device,
        initialization=initialization,
        generator=generator,
    )


def clone_parameters(parameters):
    return {name: value.detach().clone() for name, value in parameters.items()}


def _softmax(z):
    shifted = z - z.max(dim=0, keepdim=True).values
    exp_z = torch.exp(shifted)
    return exp_z / exp_z.sum(dim=0, keepdim=True)


def _batchnorm_forward(activation, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """Normalize over the batch axis of a (features, batch) activation."""
    features, batch = activation.shape
    if training and batch >= 2:
        mean = activation.mean(dim=1, keepdim=True)
        var = ((activation - mean) ** 2).mean(dim=1, keepdim=True)
        running_mean.mul_(1.0 - momentum).add_(mean.detach(), alpha=momentum)
        running_var.mul_(1.0 - momentum).add_(var.detach(), alpha=momentum)
        used_training_stats = True
    else:
        mean = running_mean
        var = running_var
        used_training_stats = False
    x_hat = (activation - mean) / torch.sqrt(var + eps)
    out = gamma * x_hat + beta
    cache = {
        "x": activation,
        "x_hat": x_hat,
        "mean": mean.detach(),
        "var": var.detach(),
        "gamma": gamma,
        "eps": eps,
        "training": used_training_stats,
    }
    return out, cache


def _batchnorm_backward(d_out, cache):
    """Backprop through batch norm. d_out is on the same scale as (softmax - Y)."""
    x = cache["x"]
    x_hat = cache["x_hat"]
    mean = cache["mean"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]
    batch = x.shape[1]
    d_gamma = torch.sum(d_out * x_hat, dim=1, keepdim=True) / batch
    d_beta = torch.sum(d_out, dim=1, keepdim=True) / batch
    d_x_hat = d_out * gamma
    inv_std = torch.rsqrt(var + eps)
    if cache["training"]:
        centered = x - mean
        d_var = torch.sum(d_x_hat * centered * (-0.5) * (var + eps) ** (-1.5), dim=1, keepdim=True)
        d_mean = torch.sum(d_x_hat * -inv_std, dim=1, keepdim=True)
        d_mean = d_mean + d_var * torch.sum(-2.0 * centered, dim=1, keepdim=True) / batch
        d_x = d_x_hat * inv_std + d_var * 2.0 * centered / batch + d_mean / batch
    else:
        d_x = d_x_hat * inv_std
    return d_x, d_gamma, d_beta


def forward(x, parameters, use_bnorm=False, training=False):
    """Return softmax probabilities (classes, batch) and per-layer caches."""
    if x.shape[0] != parameters["W1"].shape[1]:
        raise ValueError(
            f"Expected {parameters['W1'].shape[1]} features, got {x.shape[0]}."
        )
    caches = []
    activation = x
    layers = _layer_count(parameters)
    for layer in range(1, layers):
        weight = parameters[f"W{layer}"]
        bias = parameters[f"b{layer}"]
        z = weight @ activation + bias
        pre_activation = activation
        activation = torch.clamp(z, min=0.0)
        bn_cache = None
        if use_bnorm:
            activation, bn_cache = _batchnorm_forward(
                activation,
                parameters[f"gamma{layer}"],
                parameters[f"beta{layer}"],
                parameters[f"running_mean{layer}"],
                parameters[f"running_var{layer}"],
                training=training,
            )
        caches.append({
            "linear": {"A": pre_activation, "W": weight, "b": bias},
            "z": z,
            "bn": bn_cache,
        })
    weight = parameters[f"W{layers}"]
    bias = parameters[f"b{layers}"]
    z = weight @ activation + bias
    probabilities = _softmax(z)
    caches.append({
        "linear": {"A": activation, "W": weight, "b": bias},
        "z": z,
        "bn": None,
    })
    return probabilities, caches


def cost_terms(probabilities, y, parameters=None, lambd=0.0):
    """Return mean cross-entropy and the L2 term as separate tensors.

    The penalty is lambd / (2m) * sum of squared weights, where m is the column
    count of this call. A minibatch and the full validation matrix therefore do
    not put the same penalty on the same weights. Early stopping still uses the
    sum of the two terms.
    """
    if y.shape != probabilities.shape:
        raise ValueError("Labels and predictions must share a shape.")
    batch = y.shape[1]
    log_probs = torch.log(probabilities.clamp(min=1e-8))
    cross_entropy = -torch.sum(y * log_probs) / batch
    penalty = probabilities.new_zeros(())
    if lambd > 0:
        if parameters is None:
            raise ValueError("L2 cost needs the parameter dictionary.")
        weight_sum = probabilities.new_zeros(())
        for layer in range(1, _layer_count(parameters) + 1):
            weight_sum = weight_sum + torch.sum(parameters[f"W{layer}"] ** 2)
        penalty = lambd / (2.0 * batch) * weight_sum
    elif lambd < 0:
        raise ValueError("lambd must be >= 0.")
    return cross_entropy, penalty


def compute_cost(probabilities, y, parameters=None, lambd=0.0):
    """Mean cross-entropy plus the L2 term from cost_terms when lambd > 0."""
    cross_entropy, penalty = cost_terms(probabilities, y, parameters, lambd)
    return cross_entropy + penalty


def _linear_backward(d_z, linear_cache, lambd):
    activation_prev = linear_cache["A"]
    weight = linear_cache["W"]
    batch = activation_prev.shape[1]
    d_w = (d_z @ activation_prev.T) / batch
    if lambd > 0:
        d_w = d_w + (lambd / batch) * weight
    d_b = torch.sum(d_z, dim=1, keepdim=True) / batch
    d_a_prev = weight.T @ d_z
    return d_a_prev, d_w, d_b


def backward(probabilities, y, caches, lambd=0.0):
    grads = {}
    layers = len(caches)
    # Combined softmax + mean-cross-entropy gradient, before the 1/m inside dW.
    d_z = probabilities - y
    d_a_prev, d_w, d_b = _linear_backward(d_z, caches[-1]["linear"], lambd)
    grads[f"dW{layers}"] = d_w
    grads[f"db{layers}"] = d_b
    for layer in range(layers - 1, 0, -1):
        cache = caches[layer - 1]
        d_a = d_a_prev
        if cache["bn"] is not None:
            d_a, d_gamma, d_beta = _batchnorm_backward(d_a, cache["bn"])
            grads[f"dgamma{layer}"] = d_gamma
            grads[f"dbeta{layer}"] = d_beta
        d_z = d_a.clone()
        d_z[cache["z"] <= 0] = 0
        d_a_prev, d_w, d_b = _linear_backward(d_z, cache["linear"], lambd)
        grads[f"dW{layer}"] = d_w
        grads[f"db{layer}"] = d_b
    return grads


def update_parameters(parameters, grads, learning_rate):
    updated = {}
    for layer in range(1, _layer_count(parameters) + 1):
        updated[f"W{layer}"] = parameters[f"W{layer}"] - learning_rate * grads[f"dW{layer}"]
        updated[f"b{layer}"] = parameters[f"b{layer}"] - learning_rate * grads[f"db{layer}"]
        if f"gamma{layer}" in parameters:
            updated[f"gamma{layer}"] = parameters[f"gamma{layer}"] - learning_rate * grads[f"dgamma{layer}"]
            updated[f"beta{layer}"] = parameters[f"beta{layer}"] - learning_rate * grads[f"dbeta{layer}"]
            updated[f"running_mean{layer}"] = parameters[f"running_mean{layer}"]
            updated[f"running_var{layer}"] = parameters[f"running_var{layer}"]
    return updated


def predict(x, y, parameters, use_bnorm=False):
    """Accuracy on this tensor only. Batch norm uses running statistics."""
    if x.shape[1] != y.shape[1]:
        raise ValueError("Features and labels have different example counts.")
    device = parameters["W1"].device
    probabilities, _ = forward(x.to(device), parameters, use_bnorm=use_bnorm, training=False)
    predictions = torch.argmax(probabilities, dim=0)
    labels = torch.argmax(y.to(device), dim=0)
    matches = predictions == labels
    return {
        "accuracy": matches.float().mean().item(),
        "correct": int(matches.sum().item()),
        "total": int(matches.numel()),
    }


def _epoch_cost(x, y, parameters, use_bnorm, lambd, training):
    probabilities, _ = forward(x, parameters, use_bnorm=use_bnorm, training=training)
    cross_entropy, penalty = cost_terms(probabilities, y, parameters, lambd)
    cost = cross_entropy + penalty
    return (
        float(cost.detach().item()),
        float(cross_entropy.detach().item()),
        float(penalty.detach().item()),
    )


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    layer_dims,
    learning_rate=0.009,
    num_iterations=300,
    batch_size=64,
    use_bnorm=False,
    lambd=0.0,
    initialization="legacy_fan_in",
    patience=20,
    seed=21,
    device=None,
    log_every=0,
):
    """Mini-batch training. num_iterations is the maximum number of epochs.

    Early stopping watches validation cost and restores the best parameters,
    including the batch-norm running statistics from that epoch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if patience < 1:
        raise ValueError("patience must be at least 1 epoch.")
    if x_train.shape[1] != y_train.shape[1] or x_val.shape[1] != y_val.shape[1]:
        raise ValueError("A split has a different number of features and labels.")
    if x_train.shape[0] != layer_dims[0] or y_train.shape[0] != layer_dims[-1]:
        raise ValueError("layer_dims do not match the training tensors.")
    if initialization not in INITIALIZATIONS:
        raise ValueError(f"initialization must be one of {INITIALIZATIONS}.")

    device = get_device() if device is None else torch.device(device)
    torch.manual_seed(seed)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    parameters = seeded_parameters(layer_dims, use_bnorm, initialization, seed, device)

    history = {
        "train_cost": [],
        "val_cost": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_cross_entropy": [],
        "train_l2_penalty": [],
        "val_cross_entropy": [],
        "val_l2_penalty": [],
    }
    best_cost = math.inf
    best_parameters = clone_parameters(parameters)
    best_epoch = -1
    waits = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n_train = x_train.shape[1]

    for epoch in range(num_iterations):
        permutation = torch.randperm(n_train, generator=generator)
        batch_costs = []
        batch_cross_entropy = []
        batch_penalty = []
        for start in range(0, n_train, batch_size):
            columns = permutation[start:start + batch_size]
            if columns.numel() < 1:
                continue
            x_batch = x_train[:, columns]
            y_batch = y_train[:, columns]
            probabilities, caches = forward(x_batch, parameters, use_bnorm=use_bnorm, training=True)
            cross_entropy, penalty = cost_terms(probabilities, y_batch, parameters, lambd)
            cost = cross_entropy + penalty
            grads = backward(probabilities, y_batch, caches, lambd)
            parameters = update_parameters(parameters, grads, learning_rate)
            batch_costs.append(float(cost.detach().item()))
            batch_cross_entropy.append(float(cross_entropy.detach().item()))
            batch_penalty.append(float(penalty.detach().item()))

        train_cost = sum(batch_costs) / len(batch_costs)
        train_accuracy = predict(x_train, y_train, parameters, use_bnorm=use_bnorm)["accuracy"]
        val_cost, val_cross_entropy, val_penalty = _epoch_cost(
            x_val, y_val, parameters, use_bnorm, lambd, training=False
        )
        val_accuracy = predict(x_val, y_val, parameters, use_bnorm=use_bnorm)["accuracy"]
        history["train_cost"].append(train_cost)
        history["val_cost"].append(val_cost)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_cross_entropy"].append(sum(batch_cross_entropy) / len(batch_cross_entropy))
        history["train_l2_penalty"].append(sum(batch_penalty) / len(batch_penalty))
        history["val_cross_entropy"].append(val_cross_entropy)
        history["val_l2_penalty"].append(val_penalty)

        if val_cost < best_cost:
            best_cost = val_cost
            best_epoch = epoch
            best_parameters = clone_parameters(parameters)
            waits = 0
        else:
            waits += 1
            if waits >= patience:
                if log_every:
                    print(
                        f"epoch {epoch + 1} train_cost {train_cost:.4f} val_cost {val_cost:.4f} "
                        f"train_acc {train_accuracy:.4f} val_acc {val_accuracy:.4f} stop",
                        flush=True,
                    )
                break
        if log_every and (epoch + 1) % int(log_every) == 0:
            print(
                f"epoch {epoch + 1} train_cost {train_cost:.4f} val_cost {val_cost:.4f} "
                f"train_acc {train_accuracy:.4f} val_acc {val_accuracy:.4f}",
                flush=True,
            )

    history["best_epoch"] = best_epoch
    history["epochs_ran"] = len(history["train_cost"])
    history["restored_best"] = True
    history["initialization"] = initialization
    history["seed"] = int(seed)
    history["use_bnorm"] = bool(use_bnorm)
    history["lambd"] = float(lambd)
    history["selection_metric"] = "val_cost"
    return best_parameters, history


def confusion_matrix(x, y, parameters, use_bnorm=False):
    """Rows are true classes, columns are predicted classes. Uses running batch-norm stats."""
    device = parameters["W1"].device
    probabilities, _ = forward(x.to(device), parameters, use_bnorm=use_bnorm, training=False)
    predictions = torch.argmax(probabilities, dim=0).detach().cpu()
    labels = torch.argmax(y, dim=0).detach().cpu()
    index = labels * probabilities.shape[0] + predictions
    counts = torch.bincount(index, minlength=probabilities.shape[0] ** 2)
    width = int(probabilities.shape[0])
    return counts.reshape(width, width)


def _clone_leaves(parameters):
    leaves = {}
    for name, value in parameters.items():
        if name.startswith("running_"):
            leaves[name] = value.detach().clone()
        else:
            leaves[name] = value.detach().clone().requires_grad_(True)
    return leaves


def _max_grad_gap(parameters, manual, leaves):
    gaps = []
    for name, value in leaves.items():
        if value.grad is None:
            continue
        manual_name = "d" + name
        if manual_name not in manual:
            raise KeyError(f"Manual backward has no {manual_name}.")
        gaps.append(float((value.grad - manual[manual_name]).abs().max().item()))
    if not gaps:
        raise RuntimeError("Autograd produced no parameter gradients.")
    return max(gaps)


def _numeric_grad_gap(x, y, parameters, lambd, step=1e-4):
    """Central differences on a few entries. Batch norm is excluded; its check is autograd."""
    base, _ = forward(x, parameters, use_bnorm=False, training=False)
    reference = float(compute_cost(base, y, parameters, lambd).item())
    manual_probs, caches = forward(x, parameters, use_bnorm=False, training=False)
    manual = backward(manual_probs, y, caches, lambd)
    del reference
    gaps = []
    probes = [("W1", 0, 0), ("b1", 0, 0), ("W2", 0, 0)]
    for name, row, col in probes:
        original = parameters[name].detach().clone()
        manual_value = float(manual["d" + name][row, col].item())
        direction = torch.zeros_like(original)
        direction[row, col] = step
        parameters[name] = original + direction
        plus, _ = forward(x, parameters, use_bnorm=False, training=False)
        plus_cost = float(compute_cost(plus, y, parameters, lambd).item())
        parameters[name] = original - direction
        minus, _ = forward(x, parameters, use_bnorm=False, training=False)
        minus_cost = float(compute_cost(minus, y, parameters, lambd).item())
        parameters[name] = original
        numeric = (plus_cost - minus_cost) / (2.0 * step)
        gaps.append(abs(numeric - manual_value))
    return max(gaps) if gaps else 0.0


def check_gradients(seed=0, initialization="legacy_fan_in"):
    """Compare manual backward to autograd, and to finite differences when batch norm is off.

    The fixture is a few rows and a 5-4-3 network. It does not replace full MNIST training.
    initialization is applied to every case. It does not switch on with batch norm.
    """
    if initialization not in INITIALIZATIONS:
        raise ValueError(f"initialization must be one of {INITIALIZATIONS}.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    layer_dims = [5, 4, 3]
    x = torch.rand(5, 6, generator=generator)
    labels = torch.randint(0, 3, (6,), generator=generator)
    y = torch.zeros(3, 6)
    y[labels, torch.arange(6)] = 1.0
    cases = []
    for use_bnorm, lambd in ((False, 0.0), (False, 0.1), (True, 0.0), (True, 0.05)):
        torch.manual_seed(seed)
        parameters = initialize_parameters(
            layer_dims,
            use_bnorm=use_bnorm,
            device="cpu",
            initialization=initialization,
        )
        leaves = _clone_leaves(parameters)
        probabilities, _ = forward(x, leaves, use_bnorm=use_bnorm, training=True)
        loss = compute_cost(probabilities, y, leaves, lambd)
        loss.backward()
        manual_probabilities, caches = forward(x, parameters, use_bnorm=use_bnorm, training=True)
        manual = backward(manual_probabilities, y, caches, lambd)
        autograd_gap = _max_grad_gap(parameters, manual, leaves)
        numeric_gap = None if use_bnorm else _numeric_grad_gap(x, y, parameters, lambd)
        cases.append({
            "use_bnorm": use_bnorm,
            "lambd": lambd,
            "initialization": initialization,
            "max_abs_autograd": autograd_gap,
            "max_abs_numeric": numeric_gap,
        })
    return {"cases": cases, "fixture": {"examples": 6, "layer_dims": layer_dims}}


def weight_norms(parameters):
    norms = {}
    for layer in range(1, _layer_count(parameters) + 1):
        norms[f"W{layer}"] = float(torch.norm(parameters[f"W{layer}"]).item())
    return norms


if __name__ == "__main__":
    print("Import this module from experiments.ipynb. Direct execution does not download or train.")
