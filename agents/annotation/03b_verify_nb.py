"""Verify the NB logpmf in Method C matches scipy.stats.nbinom and R's dnbinom(mu) param."""
import numpy as np
from scipy.special import gammaln
from scipy.stats import nbinom

r = 10.0
for (x, mu) in [(0, 0.5), (3, 1.2), (7, 4.0), (2, 0.01)]:
    # my formula
    mine = (gammaln(x + r) - gammaln(r) - gammaln(x + 1.0)
            + r * (np.log(r) - np.log(r + mu)) + x * (np.log(mu) - np.log(r + mu)))
    # scipy: nbinom(n=r, p=r/(r+mu))
    p = r / (r + mu)
    sci = nbinom.logpmf(x, r, p)
    print(f"x={x} mu={mu}: mine={mine:.6f} scipy={sci:.6f} diff={abs(mine-sci):.2e}")
