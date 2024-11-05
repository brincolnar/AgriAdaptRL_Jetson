from torch import argmax, where, cat, stack


def get_binary_masks_infest(mask, batch_dim=True, dim=3):
    result = []
    if batch_dim:
        # Arg max each mask separately
        arg_mask = argmax(mask, dim=1)
        for i in range(mask.shape[0]):
            # Create per-class binary masks
            b_mask = where(arg_mask[i] == 0, 1, 0)[None, :, :]
            w_mask = where(arg_mask[i] == 1, 1, 0)[None, :, :]
            # If we have a lettuce class as well
            if dim == 3:
                l_mask = where(arg_mask[i] == 2, 1, 0)[None, :, :]
                result.append(cat((b_mask, w_mask, l_mask)))
            elif dim == 2:
                result.append(cat((b_mask, w_mask)))
        return stack(result)
