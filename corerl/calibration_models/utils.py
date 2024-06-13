
def new_fictitious_obs(endo_predictions, obs, endo_inds):
    fictitious_obs = obs.copy()
    for i, j in enumerate(endo_inds):
        fictitious_obs[j] = endo_predictions[i]
    return fictitious_obs