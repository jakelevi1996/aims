import torch

def cross_entropy_loss(logits, targets):
    """
    logits should be a Tensor with shape batch_size * num_classes containing
    logit predictions for each class (that will be used as the input to the
    softmax function), and targets should be a 1D Tensor with shape
    num_classes, containing the index of the correct class for each data point
    in the batch
    """
    exp_logits = torch.exp(logits)
    softmax_correct = (
        exp_logits[range(len(targets)), targets]
        / torch.sum(exp_logits, dim=1)
    )
    return -torch.mean(torch.log(softmax_correct))
