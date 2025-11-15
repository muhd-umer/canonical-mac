"""Water-filling allocation for frequency-selective channels."""

import numpy as np


def fmwaterfill_gn(gn, b_bar, gap, cb):
    """Water-filling allocation for a frequency-selective channel.

    Distributes rate and energy across subchannels using the classic water-filling
    solution while honoring a gap parameter and complex/real baseband scaling.

    Args:
        gn: Channel gains, shape (N,) array
        b_bar: Target bit rate per tone (b_total / Ntot)
        gap: Gap in dB
        cb: 1 for complex baseband, 2 for real baseband

    Returns:
        bn: Bit allocation per subchannel, shape (N,)
        en: Energy allocation per subchannel, shape (N,)
    """
    # convert db gap to linear scale
    gap_linear = 10 ** (gap / 10)

    # ensure gn is 1d array
    gn = np.asarray(gn).flatten()
    Ntot = len(gn)

    # initialization
    en = np.zeros(Ntot)
    bn = np.zeros(Ntot)

    # sort gains in descending order and keep indices
    Index = np.argsort(-gn)  # descending order
    gn_sorted = gn[Index]

    # number of zero-gain subchannels
    num_zero_gn = np.sum(gn_sorted == 0)
    Nstar = Ntot - num_zero_gn

    # find number of used channels (iteratively reduce until all energies are positive)
    while True:
        # calculate water level K
        logK = np.log(gap_linear) + (1 / Nstar) * (
            Ntot * b_bar * 2 * np.log(2) - np.sum(np.log(gn_sorted[:Nstar]))
        )
        K = np.exp(logK)

        # check if minimum energy is positive (worst channel)
        En_min = K - gap_linear / gn_sorted[Nstar - 1]

        if En_min < 0:
            Nstar = Nstar - 1  # if negative energy, use fewer channels
        else:
            break  # if all energies positive, done

    # calculate energy and bit allocations for active channels
    En = K - gap_linear / gn_sorted[:Nstar]
    Bn = (1 / cb) * np.log2(K * gn_sorted[:Nstar] / gap_linear)

    # return values in original index order
    bn[Index[:Nstar]] = Bn
    en[Index[:Nstar]] = En

    return bn, en
