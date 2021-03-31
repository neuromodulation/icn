import pandas as pd

mov_substr_list = ["TTL", "ANALOG", "MOV"]
used_substr_list = ["ECOG", "SEEG"]

def set_M1(ch_names, ch_type):

    # set here: name, reference, used, target, ECoG 
    df = pd.DataFrame(np.nan, index=np.arange(len(list(ch_names))), columns=['name', 'rereference', 'used', 'target'])

    ch_used = [ch_idx for ch_idx, ch in enumerate(ch_type) if any(used_substr in ch for used_substr in used_substr_list)]
    used = np.zeros(len(ch_names))
    used[ch_used] = 1
    df['used'] = used.astype(int)
    df['name'] = ch_names


    # check here for TTL, ANALOG 
    ch_mov = [ch_idx for ch_idx, ch in enumerate(ch_names) if any(mov_substr in ch for mov_substr in mov_substr_list)]
    target = np.zeros(len(ch_names))
    target[ch_mov] = 1
    df['target'] = target.astype(int)

    rereference = ["" for x in range(len(ch_names)))]

    for ch_idx, ch in enumerate(ch_names):
        if ch_type[ch_idx] == "ECOG":
            rereference[ch_idx]= "average"

    df['rereference'] =rereference