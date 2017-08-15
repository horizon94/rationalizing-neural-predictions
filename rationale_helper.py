from __future__ import print_function, division


def rationale_to_string(words, rationale):
    """
    rationale is a torch.LongTensor (possibly cuda-ised), containing word idxes
    we're just goign to use the words list to convert these to a string
    (words is a list of words, that we will look up the idxes in)
    """
    res = ''
    if rationale.is_cuda:
        rationale = rationale.cpu()
    # print('type(rationale)', type(rationale))
    # print('rationale.shape', rationale.shape)
    T = rationale.shape[0]
    # print('T', T)
    res_list = []
    for t in range(T):
        idx = rationale[t]
        word = words[idx]
        if word not in ['<pad>']:
            res_list.append(word)
    return ' '.join(res_list)
