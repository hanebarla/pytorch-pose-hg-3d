import numpy as np


def get_preds(hm, return_conf=False):
    """
    Estimate 2D keypoints position
    """
    assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
    h = hm.shape[2]
    w = hm.shape[3]
    hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
    idx = np.argmax(hm, axis=2)
    hmmax = np.max()
    ambiguous_idx = {}

    preds = np.zeros((hm.shape[0], hm.shape[1], 2))
    for i in range(hm.shape[0]):  # batchsize
        for j in range(hm.shape[1]):  # keypoints num
            preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
            if hm[i, j, idx[i, j]] < hmmax * 0.1:  # Ambiguous keypoints
                ambiguous_idx[(i, j)] = [idx[i, j] % w, idx[i, j] / w]
    if return_conf:
        conf = np.amax(hm, axis=2).reshape(hm.shape[0], hm.shape[1], 1)
        return preds, conf
    else:
        return preds, ambiguous_idx


def calc_dists(preds, gt, normalize):
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
                dists[j][i] = \
                    ((gt[i][j] - preds[i][j]) ** 2).sum() ** 0.5 / normalize[i]
            else:
                dists[j][i] = -1
    return dists


def dist_accuracy(dist, thr=0.5):
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum() / len(dist)
    else:
        return -1


def accuracy(output, target, acc_idxs):
    preds, amb_idx = get_preds(output)
    gt, amb_idx = get_preds(target)
    dists = calc_dists(
        preds,
        gt,
        np.ones(
            target.shape[0]) *
        target.shape[2] /
        10)
    acc = np.zeros(len(acc_idxs))
    avg_acc = 0
    bad_idx_count = 0

    for i in range(len(acc_idxs)):
        acc[i] = dist_accuracy(dists[acc_idxs[i]])
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
        else:
            bad_idx_count = bad_idx_count + 1

    if bad_idx_count == len(acc_idxs):
        return 0
    else:
        return avg_acc / (len(acc_idxs) - bad_idx_count)


def get_preds_3d(heatmap, depthmap, ambiguous_idx=None):
    ignoreidx_img = []
    output_res = max(heatmap.shape[2], heatmap.shape[3])
    preds, amb_idx = get_preds(heatmap)
    preds = preds.astype(np.int32)
    preds_3d = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.float32)
    for i in range(preds.shape[0]):  # batchsize
        ignoreidx = []
        for j in range(preds.shape[1]):  # keypoints num
            idx = min(j, depthmap.shape[1] - 1)
            pt = preds[i, j]
            try:
                amb_idx = ambiguous_idx[(i, j)]
                if pt[0] == amb_idx[0] and pt[1] == amb_idx[1]:
                    ignoreidx.append(j)
            except KeyError:
                pass
            preds_3d[i, j, 2] = depthmap[i, idx, pt[1], pt[0]]
            preds_3d[i, j, :2] = 1.0 * preds[i, j] / output_res
        ignoreidx_img.append(ignoreidx)
        preds_3d[i] = preds_3d[i] - preds_3d[i, 6:7]
    return preds_3d, ignoreidx_img


def mpjpe(heatmap, depthmap, gt_3d, convert_func):
    preds_3d, ignoreidx_img = get_preds_3d(heatmap, depthmap)
    cnt, pjpe = 0, 0
    for i in range(preds_3d.shape[0]):
        if gt_3d[i].sum() ** 2 > 0:
            cnt += 1
            pred_3d_h36m = convert_func(preds_3d[i])
            err = (((gt_3d[i] - pred_3d_h36m) ** 2).sum(axis=1) ** 0.5).mean()
            pjpe += err
    if cnt > 0:
        pjpe /= cnt
    return pjpe, cnt
