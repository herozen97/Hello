import numpy as np
import itertools



def construct_traj_true(loc_chain, tim_chain, TOKEN, traj_len):
    '''
    loc_chain, tim_chain: (S=24)
    '''

    traj_day = []
    for loc, tim in zip(loc_chain, tim_chain):
        if loc in [0, TOKEN.EOS]:
            break
        traj_day += [loc] * tim
    
    assert len(traj_day) == traj_len, 'Error in contruct_traj_true'

    return traj_day



def construct_traj_sync(loc_chain_raw, dur_chain_raw, TOKEN, traj_len, home_id):
    '''
    Input: *_chain_raw is list or array
    Output: list
    '''

    loc_chain, dur_chain = [], []
    # iterate each visit
    for idx, loc in enumerate(loc_chain_raw):
        if loc in [0, TOKEN.SOS]: continue
        if loc == TOKEN.EOS: break
        dur = dur_chain_raw[idx]
        if len(loc_chain) > 0 and loc == loc_chain[-1]:   # duplicate
            dur_chain[-1] += dur
        else:
            loc_chain.append(loc)
            dur_chain.append(dur)
    # determine meanless trajectory: stay at home
    if len(loc_chain) == 0:
        return [home_id] * traj_len

    # adjust duration
    dur_chain = np.array(dur_chain)
    inte = dur_chain.round()
    inte[inte<0] = 0
    inte[inte<1] += 1
    delta = traj_len - inte.sum().astype(int)
    if delta > 0:      # add to inte
        chain_len = len(inte)
        if delta > chain_len:   # all add and the rest add randomly
            inte += (delta // chain_len)
            delta = delta % chain_len
        for idx in np.random.choice(chain_len, delta, replace=False): inte[idx] += 1
    elif delta < 0:    # minus inte
        delta = -delta
        valid = np.where(inte > 1)[0]
        while delta >= len(valid) and len(valid)>0:   # all valid minus
            inte[inte>1] -= 1
            delta = delta - len(valid)
            valid = np.where(inte > 1)[0]
        for idx in np.random.choice(valid, delta, replace=False): inte[idx] -= 1

    if (inte < 1).any(): print('\nSMALL', dur_chain, inte, delta); raise Exception
    if (inte > traj_len).any(): print('\nLARGE', dur_chain, inte, delta); raise Exception
    if inte.sum() != traj_len: print('\nSUM', dur_chain, inte, delta); raise Exception
    dur_chain = inte.astype(int).tolist()

    # construct to trajectory
    traj_day = []
    for loc, tim in zip(loc_chain, dur_chain):
        traj_day += [loc] * tim

    if len(traj_day) != traj_len:
        raise Exception('Error in construct_traj_sync')

    return traj_day