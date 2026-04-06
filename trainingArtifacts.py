import csv
import json
import os

import numpy as np
import torch


class TrainingLogger:
    
    HEADER = ["batch", "meanReward", "policyLoss", "valueLoss", "entropy", "meanAdvantage"]

    def __init__(self, saveDir):
        self._file = None
        self._writer = None
        if saveDir is not None:
            os.makedirs(saveDir, exist_ok=True)
            self._file = open(os.path.join(saveDir, "training_log.csv"), "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self.HEADER)
            self._file.flush()

    def logBatch(self, batchIdx, meanReward, losses):
        # losses dict must contain policyLoss, valueLoss, entropy, meanAdvantage
        if self._writer is None:
            return
        self._writer.writerow([
            batchIdx + 1,
            f"{meanReward:.6f}",
            f"{losses['policyLoss']:.6f}",
            f"{losses['valueLoss']:.6f}",
            f"{losses['entropy']:.6f}",
            f"{losses['meanAdvantage']:.6f}",
        ])
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def saveNNCheckpoint(actor, critic, saveDir, tag="final"):
    # Saves actor and critic state dicts.
    # tag: suffix for the filename, e.g. "final" or "batch_100"
    os.makedirs(saveDir, exist_ok=True)
    torch.save(actor.state_dict(),  os.path.join(saveDir, f"actor_{tag}.pth"))
    torch.save(critic.state_dict(), os.path.join(saveDir, f"critic_{tag}.pth"))


def loadNNCheckpoint(actor, critic, saveDir, tag="final"):
    # Loads actor and critic state dicts in-place.
    actor.load_state_dict( torch.load(os.path.join(saveDir, f"actor_{tag}.pth"),  weights_only=True))
    critic.load_state_dict(torch.load(os.path.join(saveDir, f"critic_{tag}.pth"), weights_only=True))


def saveLQIModel(K, Q, R, stepInfo, saveDir):
    # Saves LQI gain matrix K and cost matrices Q, R as a .npz file.
    # stepInfo dict (RiseTime, SettlingTime, Overshoot, ...) is saved as JSON.
    os.makedirs(saveDir, exist_ok=True)
    np.savez(os.path.join(saveDir, "lqi_model.npz"), K=K, Q=Q, R=R)
    with open(os.path.join(saveDir, "lqi_step_info.json"), "w") as f:
        json.dump({k: float(v) for k, v in stepInfo.items()}, f, indent=2)


def loadLQIModel(saveDir):
    # Returns (K, Q, R, stepInfo).
    data = np.load(os.path.join(saveDir, "lqi_model.npz"))
    with open(os.path.join(saveDir, "lqi_step_info.json")) as f:
        stepInfo = json.load(f)
    return data["K"], data["Q"], data["R"], stepInfo
