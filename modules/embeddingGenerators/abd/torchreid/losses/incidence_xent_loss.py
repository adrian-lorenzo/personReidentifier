from modules.abd.torchreid.utils.environ import get_env_or_raise

from .cross_entropy_loss import CrossEntropyLoss
from .incidence_loss import IncidenceLoss


def IncidenceXentLoss(num_classes, use_gpu=True, label_smooth=True):
    beta = get_env_or_raise(float, 'incidence_beta', 'beta')
    incidence_loss = IncidenceLoss()
    xent_loss = CrossEntropyLoss(num_classes, use_gpu=use_gpu, label_smooth=label_smooth)

    def _loss(x, pids):
        return (
                xent_loss(x[1], pids) +
                beta * incidence_loss(x, pids)
        )

    return _loss
