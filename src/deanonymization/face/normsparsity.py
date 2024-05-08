from .abstract import AbstractFaceDeanonymization

import os
import matlab.engine


class NormsparsityDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces by Blind deconvolution using a normalized sparsity measure

    Requires installed matlab and matlab pip.
    Requires patched paper implementation from in `bin/norm_sparsity` (patch in scripts/norm_sparsity.patch)

    NOTE: If you receive errors "free(): invalid pointer Aborted (core dumped)" after "Deanonymization successful",
          you can rerun the configuration and caching will use the correct dataset. This seems to be an issue with matlab pip.

    Paper:
        Krishnan, D., Tay, T. and Fergus, R., 2011, June.
        Blind deconvolution using a normalized sparsity measure. In CVPR 2011 (pp. 233-240). IEEE.

    Required pips:
        - matlab

    Parameters:
        none
    """

    name = "normsparsity"

    def deanonymize_all(self):
        eng = matlab.engine.start_matlab()
        eng.addpath(os.path.join(os.getcwd(), "bin", "norm_sparsity"))

        for point in self.dataset.datapoints.values():
            old, new = eng.test_blind_deconv(point.get_path(), nargout=2)
            eng.imwrite(new, point.get_path(), nargout=0)

        eng.quit()
