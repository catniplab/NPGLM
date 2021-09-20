from collections import OrderedDict

import torch
import numpy as np

from NPGLM_v2p0.glm import covariate
from NPGLM_v2p0.glm import npglm


def main():
    X = OrderedDict()
    X['c1'] = torch.randn(10, 100)
    Y = torch.randn(10, 100)

    c1 = covariate.Covariate('c1', 1e-3, 0.0, 1)
    glm = npglm.NPGLM()

    glm.add_covariate(c1)
    glm.forward(X, Y)


if __name__ == '__main__':
    main()
