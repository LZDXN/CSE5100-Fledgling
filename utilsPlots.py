# Common utils file for plot code, etc.

# Global libraries
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os


# Local project libraries


class LivePlotter:
    # Two live-updating figures that refresh during training:
    #   Figure 1 -- reward curve (mean reward per batch + rolling mean)
    #   Figure 2 -- altitude tracking on top, per-rotor thrust grid below
    # Every update() call also saves both figures to saveDir/plots/.

    def __init__(self, nRotors=4, hoverPct=0.5, saveDir=None):
        plt.ion()
        self.nRotors = nRotors
        self.hoverPct = hoverPct

        if saveDir is not None:
            self.plotDir = os.path.join(saveDir, "plots")
            os.makedirs(self.plotDir, exist_ok=True)
        else:
            self.plotDir = None

        nRows = int(np.ceil(np.sqrt(nRotors)))
        nCols = int(np.ceil(nRotors / nRows))
        self.nRotorRows = nRows
        self.nRotorCols = nCols

        # -- Graph 1: reward curve --
        self.fig1, self.axReward = plt.subplots(figsize=(7, 4))
        self.fig1.suptitle("Training Reward")
        self.axReward.set_xlabel("Batch")
        self.axReward.set_ylabel("Mean Reward")
        self.axReward.grid()
        self.fig1.tight_layout()

        # -- Graph 2: altitude (top, full width) + rotor grid (bottom) --
        self.fig2 = plt.figure(
            figsize=(4 * nCols, 3 * (nRows + 1)), constrained_layout=True
        )
        gs = GridSpec(nRows + 1, nCols, figure=self.fig2)

        self.axAlt = self.fig2.add_subplot(gs[0, :])
        self.axAlt.set_title("Tracking (Eval)")
        self.axAlt.set_xlabel("Time (s)")
        self.axAlt.set_ylabel("Position")
        self.axAlt.grid()

        self.axRotors = []
        for i in range(nRotors):
            r, c = divmod(i, nCols)
            ax = self.fig2.add_subplot(gs[r + 1, c])
            ax.set_title(f"Rotor {i + 1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Thrust")
            ax.grid()
            self.axRotors.append(ax)

        # Hide any leftover grid cells
        for j in range(nRotors, nRows * nCols):
            r, c = divmod(j, nCols)
            self.fig2.add_subplot(gs[r + 1, c]).set_visible(False)

        plt.pause(0.01)

    def update(self, batchRewards, evalTraj, refCmd, stepMetrics=None):
        # -- Reward curve --
        self.axReward.cla()
        batches = np.arange(1, len(batchRewards) + 1)
        rewards = np.array(batchRewards)
        self.axReward.plot(
            batches, rewards, color="steelblue", alpha=0.4, label="reward"
        )
        if len(rewards) >= 10:
            kernel = np.ones(10) / 10
            rolling = np.convolve(rewards, kernel, mode="valid")
            self.axReward.plot(
                batches[9:],
                rolling,
                color="steelblue",
                lw=2,
                label="rolling mean (w=10)",
            )
        self.axReward.set_xlabel("Batch")
        self.axReward.set_ylabel("Mean Reward")
        self.axReward.set_title("Training Reward")
        self.axReward.legend()
        self.axReward.grid()
        self.fig1.tight_layout()
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()

        # -- Position tracking --
        self.axAlt.cla()
        self.axAlt.plot(
            evalTraj["time"], evalTraj["pos"], color="steelblue", label="pos"
        )
        self.axAlt.axhline(
            refCmd, color="tomato", linestyle="--", label=f"Target {refCmd}"
        )
        if stepMetrics is not None:
            def _fmt(v):
                return f"{v:.3f}" if not np.isnan(v) else "N/A"
            metricsLine = (
                f"Tr={_fmt(stepMetrics['riseTime'])}s  "
                f"Ts={_fmt(stepMetrics['settlingTime'])}s  "
                f"OS={stepMetrics['overshoot']:.2f}%  "
                f"US={stepMetrics['undershoot']:.2f}%"
            )
            self.axAlt.set_title(f"Tracking (Eval)\n{metricsLine}")
        else:
            self.axAlt.set_title("Tracking (Eval)")
        self.axAlt.set_xlabel("Time (s)")
        self.axAlt.set_ylabel("Position")
        self.axAlt.legend()
        self.axAlt.grid()

        # -- Per-rotor thrust --
        u = evalTraj.get("u")
        t = evalTraj["time"]
        for i, ax in enumerate(self.axRotors):
            ax.cla()
            if u is not None and i < u.shape[0]:
                ax.plot(t, u[i], color=f"C{i}")
            ax.axhline(
                self.hoverPct,
                color="k",
                linestyle="--",
                alpha=0.5,
                label=f"Hover ({self.hoverPct})",
            )
            ax.set_title(f"Rotor {i + 1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Thrust")
            ax.legend(fontsize=7)
            ax.grid()

        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

        plt.pause(0.05)

        # -- Save to disk for progress report --
        if self.plotDir is not None:
            batchNum = len(batchRewards)
            self.fig1.savefig(
                f"{self.plotDir}/reward_batch{batchNum:04d}.png",
                dpi=150,
                bbox_inches="tight",
            )
            self.fig2.savefig(
                f"{self.plotDir}/tracking_batch{batchNum:04d}.png",
                dpi=150,
                bbox_inches="tight",
            )

    def saveFinal(self):
        # Save final state of both figures with a fixed name for easy reference.
        if self.plotDir is not None:
            self.fig1.savefig(
                f"{self.plotDir}/reward_final.png", dpi=150, bbox_inches="tight"
            )
            self.fig2.savefig(
                f"{self.plotDir}/tracking_final.png", dpi=150, bbox_inches="tight"
            )

    def saveBest(self):
        # Save current figure state as the best-model eval plot.
        if self.plotDir is not None:
            self.fig1.savefig(
                f"{self.plotDir}/reward_best.png", dpi=150, bbox_inches="tight"
            )
            self.fig2.savefig(
                f"{self.plotDir}/tracking_best.png", dpi=150, bbox_inches="tight"
            )


def plotStepResponse(trajectory, title, savePath):
    # 3-panel plot: altitude, velocity, control input vs time
    # Used by humanLQI and for static saves -- not live

    time = trajectory["time"]
    z = trajectory["z"]
    zDot = trajectory["z_dot"]
    u = trajectory.get("u", np.array([]))
    zCmd = trajectory.get("z_cmd", None)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(title)

    axes[0].plot(time, z, color="steelblue", label="z (m)")
    if zCmd is not None:
        axes[0].axhline(zCmd, color="tomato", linestyle="--", label=f"Target {zCmd} m")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(time, zDot, color="darkorange", label="z_dot (m/s)")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend()
    axes[1].grid()

    if len(u) > 0:
        axes[2].plot(time[: len(u)], u, color="seagreen", label="Control input")
    axes[2].set_ylabel("Control input")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid()

    fig.tight_layout()
    fig.savefig(savePath, dpi=150)
    plt.close(fig)


def plotLqrVsRl(lqrTraj, rlTraj, zCmd, savePath):
    # Side-by-side altitude comparison: LQI baseline vs RL controller

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"LQI vs RL -- Target z = {zCmd} m")

    for ax, traj, label, color in zip(
        axes,
        [lqrTraj, rlTraj],
        ["LQI Baseline", "RL Controller (PPO)"],
        ["steelblue", "seagreen"],
    ):
        ax.plot(traj["time"], traj["z"], color=color, label=label)
        ax.axhline(
            zCmd,
            color="tomato",
            linestyle="--",
            linewidth=0.9,
            label=f"Target {zCmd} m",
        )
        ax.set_ylabel("Altitude (m)")
        ax.set_xlabel("Time (s)")
        ax.set_title(label)
        ax.legend()
        ax.grid()

    fig.tight_layout()
    fig.savefig(savePath, dpi=150)
    plt.close(fig)


# Local project libraries

# plt.rcParams["text.usetex"] = True
# plt.rcParams["figure.constrained_layout.use"] = True
