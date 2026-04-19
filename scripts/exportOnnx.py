# Export a trained actor to ONNX + dump the axis plant matrices as JSON so the
# frontend can run the sim in-browser.
#
# Usage:
#   python scripts/exportOnnx.py \
#     --run_dir data/2026-04-18_22-14-44_test \
#     --axis VERT \
#     --out_dir web_assets/
#
# Produces:
#   <out_dir>/actor_<axis>.onnx     -- tiny MLP (~few KB)
#   <out_dir>/plant_<axis>.json     -- A, B, C, E, hoverPct, obsDim, actionDim, dt

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import multiRotorPlant
import nnController


AXIS_MAP = {
    "PITCHLON": multiRotorPlant.axisEnum_enumClass.PITCHLON,
    "ROLLLAT": multiRotorPlant.axisEnum_enumClass.ROLLLAT,
    "VERT": multiRotorPlant.axisEnum_enumClass.VERT,
    "YAWHDG": multiRotorPlant.axisEnum_enumClass.YAWHDG,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="experiment dir containing <axis>/actor_best.pth")
    p.add_argument("--axis", required=True, choices=list(AXIS_MAP.keys()))
    p.add_argument("--out_dir", default="web_assets")
    p.add_argument("--n_rotors", type=int, default=4)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--hidden_dim", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    axisEnum = AXIS_MAP[args.axis]
    plant = multiRotorPlant.multiRotor6DOFWithXYZPositionError_class(
        rotorCount_nr_int=args.n_rotors
    )
    A, B, C, E = plant.discreteLQIPlant(plant.plantAxisHandler(axisEnum), args.dt)

    obsLen = int(np.sum(np.count_nonzero(C, axis=1)))

    axisDirName = args.axis.capitalize()
    candidates = [
        os.path.join(args.run_dir, axisDirName, "actor_best.pth"),
        os.path.join(args.run_dir, axisDirName, "actor_final.pth"),
        os.path.join(args.run_dir, "actor_best.pth"),
        os.path.join(args.run_dir, "actor_final.pth"),
    ]
    ckptPath = next((p for p in candidates if os.path.exists(p)), None)
    if ckptPath is None:
        raise FileNotFoundError(
            f"no actor checkpoint found; tried: {candidates}"
        )

    # Infer action_dim from the checkpoint's last linear layer; older single-action
    # runs were trained with action_dim=1, current main uses action_dim=n_rotors.
    state = torch.load(ckptPath, weights_only=True)
    actionDim = int(state["net.4.weight"].shape[0])

    actor = nnController.ActorMLP(
        obs_dim=obsLen, action_dim=actionDim, hidden=args.hidden_dim
    )
    actor.load_state_dict(state)
    actor.eval()

    # Export the deterministic mean-network only (no log_std, no sampling).
    # Frontend will clamp the output to [-1, 1] itself.
    exportNet = actor.net

    dummyInput = torch.zeros(1, obsLen, dtype=torch.float32)
    onnxPath = os.path.join(args.out_dir, f"actor_{args.axis}.onnx")
    torch.onnx.export(
        exportNet,
        dummyInput,
        onnxPath,
        input_names=["obs"],
        output_names=["action_mean"],
        dynamic_axes={"obs": {0: "batch"}, "action_mean": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {onnxPath}")

    plantJson = {
        "axis": args.axis,
        "nRotors": args.n_rotors,
        "dt": args.dt,
        "hoverPct": float(plant.rotorHoverThrustPercent_fthov_float),
        "obsDim": obsLen,
        "actionDim": actionDim,
        "stateDim": int(A.shape[0]),
        "A": np.array(A).tolist(),
        "B": np.array(B).tolist(),
        "C": np.array(C).tolist(),
        "E": np.array(E).tolist(),
    }
    plantPath = os.path.join(args.out_dir, f"plant_{args.axis}.json")
    with open(plantPath, "w") as f:
        json.dump(plantJson, f, indent=2)
    print(f"wrote {plantPath}")


if __name__ == "__main__":
    main()
