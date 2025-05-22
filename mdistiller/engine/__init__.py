from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, DEEPKD, DEEPKD_aug, AugTrainer, CRLDTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "deepkd": DEEPKD,
    "deepkd_aug": DEEPKD_aug,
    "aug": AugTrainer,
    "crld": CRLDTrainer
}
