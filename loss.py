import torch
import torch.nn.functional as F


def localization_loss(bbox_preds, bbox_targets, labels_mask):
    pos_mask = (labels_mask == 1)
    num_pos = pos_mask.sum().clamp(min=1).float()

    pred = bbox_preds[pos_mask]
    target = bbox_targets[pos_mask]

    loss = F.smooth_l1_loss(pred, target, reduction="sum")
    return loss / num_pos


def confidence_loss(cls_logits, cls_targets, labels_mask, neg_pos_ratio=3):
    """
    cls_logits: [N, C+1]
    cls_targets: [N]
    labels_mask: [N] (1=pos, 0=neg, -1=ignore)
    """

    device = cls_logits.device

    # -------------------------
    # Cross entropy for all valid anchors
    # -------------------------
    valid_mask = labels_mask != -1
    loss_all = torch.zeros_like(cls_targets, dtype=torch.float32, device=device)

    loss_all[valid_mask] = F.cross_entropy(
        cls_logits[valid_mask],
        cls_targets[valid_mask],
        reduction="none"
    )

    # -------------------------
    # Hard Negative Mining
    # -------------------------
    pos_mask = labels_mask == 1
    neg_mask = labels_mask == 0

    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()

    if num_pos == 0:
        return loss_all[pos_mask].sum()  # edge case

    max_neg = min(neg_pos_ratio * num_pos, num_neg)

    neg_losses = loss_all.clone()
    neg_losses[~neg_mask] = -1

    _, idx = torch.sort(neg_losses, descending=True)

    hard_neg_mask = torch.zeros_like(labels_mask, dtype=torch.bool)
    hard_neg_mask[idx[:max_neg]] = True

    final_mask = pos_mask | hard_neg_mask

    conf_loss = loss_all[final_mask].sum()
    conf_loss = conf_loss / max(num_pos, 1)

    return conf_loss


def ssd_loss(
    cls_logits,
    bbox_preds,
    cls_targets,
    bbox_targets,
    labels_mask,
    neg_pos_ratio=3
):
    B, A, C_plus_1 = cls_logits.shape

    cls_logits = cls_logits.view(-1, C_plus_1)
    bbox_preds = bbox_preds.view(-1, 4)
    cls_targets = cls_targets.view(-1)
    bbox_targets = bbox_targets.view(-1, 4)
    labels_mask = labels_mask.view(-1)

    loc_loss = localization_loss(
        bbox_preds,
        bbox_targets,
        labels_mask
    )

    conf_loss = confidence_loss(
        cls_logits,
        cls_targets,
        labels_mask,
        neg_pos_ratio
    )

    total_loss = loc_loss + conf_loss
    return total_loss, loc_loss, conf_loss
