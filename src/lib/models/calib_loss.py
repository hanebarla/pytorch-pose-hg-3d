import torch


def ImgLoss(output):
    h_out = output['hm']
    label = torch.zeros_like(h_out)

    for i in range(h_out.size()[0]):  # batchsize
        for j in range(h_out.size()[1]):  # keypoints num
            max_idx = torch.argmax(h_out[i, j, :, :]).item()
            r_max, c_max = max_idx // h_out.size()[3], max_idx % h_out.size()[3]
            label[i, j, r_max, c_max] = 1.0

    hm_loss = torch.nn.MSELoss()(h_out, label)

    img_loss = hm_loss + Clibloss(output)

    return img_loss


def Clibloss(output):
    d_out = output['depth']
    lab_zeros = torch.zeros_like(d_out)

    return torch.nn.MSELoss()(d_out, lab_zeros)
