import torch
import numpy as np
import scipy.optimize   # linear_sum_assignment


def adjusted_rand_index(true_mask, pred_mask, exclude_background=True):
    """
    compute the ARI for a single image. N.b. ARI 
    is invariant to permutations of the cluster IDs.

    See https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index.

    true_mask: LongTensor of shape [N, num_entities, 1, H, W]
        background == 0
        object 1 == 1
        object 2 == 2
        ...
    pred_mask: FloatTensor of shape [N, K, 1, H, W]  (mask probs)

    Returns: ari [N]

    """
    if len(true_mask.shape) == 5:
        N, max_num_entities, _, H, W = true_mask.shape
        true_groups = max_num_entities
        # only need one channel
        #true_mask = true_mask[:,0]  # [N, H, W]
        # convert into oh  [N, num_points, max_num_entities]
        true_mask = true_mask.permute(0,2,3,4,1).contiguous()
        for i in range(max_num_entities):
            mask = (true_mask[...,i] == 255)
            true_mask[...,i][mask] = i
        true_mask, _ = torch.max(true_mask, dim=-1)
    # elif len(true_mask.shape) == 4:
    #     N, C, H, W = true_mask.shape
    #     if C > 1:
    #         true_mask = true_mask[:,0]
    #     #true_groups = max_num_entities
    #     #assert true_groups > 0

    true_group_ids = true_mask.view(N, H * W).long()
    true_mask_oh = torch.nn.functional.one_hot(true_group_ids).float()
    # exclude background
    if exclude_background:
        true_mask_oh[...,0] = 0

    # take argmax across slots for predicted masks
    pred_mask = pred_mask.squeeze(2)  # [N, K, H, W]
    pred_groups = pred_mask.shape[1]
    pred_mask = torch.argmax(pred_mask, dim=1)  # [N, H, W]
    pred_group_ids = pred_mask.view(N, H * W).long()
    pred_group_oh = torch.nn.functional.one_hot(pred_group_ids, pred_groups).float()
    
    n_points = H*W

    if n_points <= max_num_entities and n_points <= pred_groups:
        
        
        raise ValueError(
                "adjusted_rand_index requires n_groups < n_points. We don't handle "
                "the special cases that can occur when you have one cluster "
                "per datapoint")

    n_points = torch.sum(true_mask_oh, dim=[1,2])  # [N]
    nij = torch.einsum('bji,bjk->bki', pred_group_oh, true_mask_oh)
    a = torch.sum(nij, 1)
    b = torch.sum(nij, 2)

    rindex = torch.sum(nij * (nij - 1), dim=[1,2])
    aindex = torch.sum(a * (a - 1), 1)
    bindex = torch.sum(b * (b - 1), 1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    # check if both single cluster; nij matrix has only 1 nonzero entry
    check_single_cluster = torch.sum( (nij > 0).int(), dim=[1,2])  # [N]
    check_single_cluster = (1 == check_single_cluster).int()
    ari[ari != ari] = 0  # remove Nan
    ari = check_single_cluster * torch.ones_like(ari) + (1 - check_single_cluster) * ari

    return ari


def pixel_mse(true_image, colors, masks):
    """
    true_image is a FloatTensor of shape [N, C, H, W]
    colors is a FloatTensor of shape [N, K, C, H, W]
    masks is a FloatTensor of shape [N, K, 1, H, W]
    """
    pred_image = torch.sum(masks * colors, 1)
    mse = torch.mean((true_image - pred_image) ** 2, dim=[1,2,3])
    return mse
    

def matching_iou(true_mask, pred_mask, presence):
    """
    true_mask: LongTensor of shape [num_entities, 1, H, W]
        background == 0
        object 1 == 1
        object 2 == 2
        ...
    pred_mask: [K,1,H,W]
    prescence: [max # of objects]  binary mask with 1's left-to-right. First index for BG (ignore).
    """
    true_mask = true_mask.squeeze(1)
    max_num_entities, H, W = true_mask.shape
    true_mask = true_mask.permute(1,2,0).contiguous()
    for i in range(max_num_entities):
        mask = (true_mask[...,i] == 255)
        true_mask[...,i][mask] = i
    true_mask, _ = torch.max(true_mask, dim=-1)
 
    pred_mask = pred_mask.squeeze(1)
    K,_,_ = pred_mask.shape
    #pred_mask_ = pred_mask.clone()
    pred_mask = torch.argmax(pred_mask, dim=0)  # [H, W]
    pred_mask = pred_mask.view(H*W).long()
    #true_mask = true_mask[:,0]  # [N, H, W]
    true_object_pixels = true_mask.view(H*W).long()
    pixel_ids = torch.arange(H*W)
    #num_objects = true_object_pixels.max() # (1,)
    num_objects = presence.sum()-1  # subtract 1 for BG
    iou = torch.zeros(int(num_objects),K)

    #for batch_id in range(N):
    remove = []
    for i in range(1, presence.shape[0]):
        viz = presence[i]
        if viz == 0:
            continue
        true_object = pixel_ids[(true_object_pixels == i)]
        if len(true_object) < 50:  # skip fully occluded or objects cropped out
            iou[i-1] = -1000000   # -INF
            remove += [i-1]
            continue
        for j in range(K):
            pred_object = pixel_ids[(pred_mask == j)]
            intersection = np.intersect1d(true_object, pred_object).shape[0]
            union = len(torch.cat([true_object,pred_object]).unique(sorted=False))
            if union == 0:
                iou[i-1,j] = -1000000
            else:
                iou[i-1,j] = intersection / union
    # NN matching
    # take max over columns (per groundtruth object) and average
    #best_iou = torch.max(iou, dim=2)[0]  # [N,num_objects]
    # Do linear assignment, which gets us a # [N, num_objects, num_objects] matrix of matches
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(iou.data.cpu().numpy(), maximize=True)
    assert row_ind.shape[0] == num_objects
    assert col_ind.shape[0] == num_objects

    for i in range(row_ind.shape[0]):
        the_iou = iou[row_ind[i], col_ind[i]]
        if the_iou < 0.1:
            remove += [i]

    if len(remove) > 0:
        row_ind = np.delete(row_ind, np.array(remove), 0)
        col_ind = np.delete(col_ind, np.array(remove), 0)
    
    # GT object indices (should always be sorted, ignores BG)
    # col_ind will index into K
    row_ind += 1
            
    return row_ind, col_ind


if __name__ == '__main__':

    true_mask = torch.LongTensor([[1, 1, 1],[0,2,2],[0,0,0]])
    true_mask = true_mask.view(1,1,3,3)

    print(true_mask)

    pred_mask = torch.FloatTensor([[[0., 0., 1.], [1., 0., 0],[1,1,1]],
            [[1., 1., 0.], [0,0,0], [0,0,0]],
            [[0,0,0], [0,1,1], [0,0,0]]])

    pred_mask = pred_mask.view(1,3,1,3,3)

    print(pred_mask)

    print(adjusted_rand_index(true_mask, pred_mask))
