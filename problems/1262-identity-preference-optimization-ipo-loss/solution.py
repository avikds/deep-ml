import numpy as np

def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    chosen_logratios = np.asarray(policy_chosen_logps) - np.asarray(ref_chosen_logps)
    rejected_logratios = np.asarray(policy_rejected_logps) - np.asarray(ref_rejected_logps)

    margin = chosen_logratios - rejected_logratios
    target = 1.0 / (2.0 * beta)

    loss = np.mean((margin - target) ** 2)
    return round(float(loss), 6)