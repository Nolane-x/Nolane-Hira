"""Explicit OOD / abstention authority.

Classifier confidence is a *signal*, not the OOD authority. R5 CLINC150 experiments show a
validation-tuned sparse max-score gate still rejects many legitimate in-scope requests while
missing a material fraction of OOS examples. Production automation must therefore require an
OOD-specific calibrated score or fall back to escalation.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class OODAction(str,Enum):
    ACCEPT='accept'
    ABSTAIN='abstain'
    ESCALATE='escalate'

@dataclass(frozen=True)
class OODReceipt:
    action:OODAction
    ood_score:float|None
    threshold:float|None
    authority:str
    calibrator_id:str|None
    reason:str

@dataclass
class OODGate:
    threshold:float
    calibrator_id:str
    authority:str='ood_calibrator'
    def decide(self,ood_score:float|None,*,distribution_shift:bool=False,calibrator_valid:bool=True)->OODReceipt:
        if distribution_shift:
            return OODReceipt(OODAction.ESCALATE,ood_score,self.threshold,self.authority,self.calibrator_id,'distribution_shift')
        if not calibrator_valid or ood_score is None:
            return OODReceipt(OODAction.ESCALATE,ood_score,self.threshold,self.authority,self.calibrator_id,'missing_or_invalid_ood_calibration')
        if not 0<=ood_score<=1: raise ValueError('ood_score must be in [0,1]')
        if ood_score>=self.threshold:
            return OODReceipt(OODAction.ABSTAIN,ood_score,self.threshold,self.authority,self.calibrator_id,'ood_score_above_threshold')
        return OODReceipt(OODAction.ACCEPT,ood_score,self.threshold,self.authority,self.calibrator_id,'ood_score_below_threshold')

def weak_classifier_confidence_signal(*,max_class_probability:float)->float:
    """Weak feature only; callers must not treat this return value as calibrated OOD probability."""
    if not 0<=max_class_probability<=1: raise ValueError('probability must be in [0,1]')
    return 1.0-max_class_probability
