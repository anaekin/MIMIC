import torch
from torchvision import models


verifier_model = models.mobilenet_v2(weights="IMAGENET1K_V1").to("cpu").eval()


def compute_accuracy(
    inputs,
    target_class,
    topk=[1, 5],
):
    """Perform validation using mobilenet_v2 model."""
    inputs = inputs.to("cpu", dtype=torch.float32)
    targets = torch.LongTensor([target_class] * inputs.shape[0]).to("cpu")

    def accuracy(output, targets, topk=[1]):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        outputs = verifier_model(inputs)
        prec = accuracy(outputs, targets, topk)
        prec_dict = {f"top{k}": p.item() for k, p in zip(topk, prec)}

    return prec_dict
