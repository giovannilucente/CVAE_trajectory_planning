import os
import logging
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Dataset(Dataset):
    def __init__(self, data_root, image_transform=None, history=3, rollout=3, dt=1, max_points=10, max_samples=None):
        self.data_root = data_root
        self.image_root = os.path.join(data_root, "imgs")
        self.image_transform = image_transform
        self.history = history
        self.rollout = rollout
        self.dt = dt
        self.max_points = max_points
        self.max_samples = max_samples

        conditions = pd.read_parquet(os.path.join(data_root, "conditions.parquet"))
        trajectories = pd.read_parquet(os.path.join(data_root, "sampled_vars.parquet"))

        self.conditions = {
            (row.scenario, row.time_step): (row.ref_x, row.ref_y)
            for row in conditions.itertuples()
        }

        self.trajectories = {
            (row.scenario, row.time_step): (row.x, row.y)
            for row in trajectories.itertuples()
        }

        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        
        for episode in tqdm(sorted(os.listdir(self.image_root)), desc="Building dataset"):
            episode_dir = os.path.join(self.image_root, episode)
            if not os.path.isdir(episode_dir):
                continue

            frames = sorted(
                int(os.path.splitext(f)[0])
                for f in os.listdir(episode_dir)
                if f.endswith(".png")
            )
            available = set(frames)

            for t in frames:
                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples

                history_frames = [t - i * self.dt for i in reversed(range(self.history))]
                future_frames  = [t + i * self.dt for i in range(1, self.rollout + 1)]

                sample_frames = history_frames + future_frames
                action_frames = [t + i * self.dt for i in range(self.rollout)]

                if not set(sample_frames).issubset(available):
                    continue
                if not all((episode, f) in self.conditions for f in action_frames):
                    continue
                if not all((episode, f) in self.trajectories for f in action_frames):
                    continue

                valid = True

                for f in action_frames:
                    ref_x, ref_y = self.conditions[(episode, f)]
                    ref_points = self._resample_path(self._get_points(ref_x, ref_y))

                    if len(ref_points) < self.max_points:
                        valid = False
                        break
                    
                    traj_x, traj_y = self.trajectories[(episode, f)]
                    traj_points = self._get_points(traj_x, traj_y)

                    if len(traj_points) < self.max_points:
                        valid = False
                        break

                if not valid:
                    continue

                samples.append((episode, sample_frames, action_frames))


        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, episode, frame):
        path = os.path.join(self.image_root, episode, f"{frame}.png")
        image = Image.open(path).convert("L")
        return self.image_transform(image) if self.image_transform else image

    def _get_points(self, x, y):
        points = list(zip(x, y))
        return torch.tensor(points, dtype=torch.float32)
    
    def _resample_path(self, points, ds=2.0):
        distances = torch.cat([torch.zeros(1), torch.cumsum(torch.norm(points[1:] - points[:-1], dim=1), dim=0)])

        target_distances = torch.arange(0, distances[-1], ds, dtype=points.dtype)
        idx = torch.searchsorted(distances, target_distances).clamp(1, len(points) - 1)

        d0 = distances[idx - 1]
        d1 = distances[idx]
        p0 = points[idx - 1]
        p1 = points[idx]

        alpha = ((target_distances - d0) / (d1 - d0)).unsqueeze(1)

        return p0 + alpha * (p1 - p0)
    
    def _to_deltas(self, points):
        previous = torch.cat([torch.zeros(1, 2), points[:-1]], dim=0)
        return points - previous

    def __getitem__(self, idx):
        episode, sample_frames, action_frames = self.samples[idx]

        frames = torch.cat([self._load_image(episode, f) for f in sample_frames], dim=0)

        reference_paths = torch.stack([
            self._to_deltas(
                self._resample_path(
                    self._get_points(*self.conditions[(episode, f)])
                )[:self.max_points]
            )
            for f in action_frames
        ])

        trajectories = torch.stack([
            self._to_deltas(
                self._get_points(*self.trajectories[(episode, f)])[:self.max_points]
            )
            for f in action_frames
        ])

        return {
            "frames": frames,                       # (history + rollout, H, W)
            "reference_paths": reference_paths,     # (rollout, len_reference_path, 2)
            "trajectories": trajectories,           # (rollout, max_points, 2)
        }


if __name__ == "__main__":
    import os
    import torch
    import matplotlib.pyplot as plt
    from torchvision import transforms

    img_dim = 256
    BASE_DIR = "../../../../mnt/nfs-share/AI_Datasets/_unzipped/world_model"
    DATA_ROOT = os.path.join(BASE_DIR, "validation")
    rollout = 3

    image_transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
    ])

    dataset = Dataset(
        data_root=DATA_ROOT,
        image_transform=image_transform,
        history=3,
        rollout=2,
        dt=5,
        max_points=20,
        max_samples=100,
    )

    sample = dataset[0]
    frames = sample["frames"]
    trajectories = sample["trajectories"]
    reference_paths = sample["reference_paths"]

    print(f"Dataset size: {len(dataset)}")
    print(f"Frames:          {frames.shape}")
    print(f"Reference paths: {reference_paths.shape}")
    print(f"Trajectories:    {trajectories.shape}")

    os.makedirs("test", exist_ok=True)

    # First 3 frames: t-2, t-1, t
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(3):
        axes[i].imshow(frames[i], cmap="gray")
        axes[i].set_title(f"Frame t-{2-i}" if i < 2 else "Frame t")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("test/images.png", dpi=150, bbox_inches="tight")
    plt.close()

    # First reference path and trajectory: at t
    reference_path = torch.cumsum(reference_paths[0], dim=0)
    trajectory = torch.cumsum(trajectories[0], dim=0)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        reference_path[:, 0],
        reference_path[:, 1],
        marker="*",
        label="Reference path",
        linewidth=2,
    )

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker="*",
        label="Future trajectory",
        linewidth=2,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Reference Path vs Future Trajectory")
    ax.axis("equal")
    ax.grid()
    ax.legend()

    plt.tight_layout()
    plt.savefig("test/paths.png", dpi=150, bbox_inches="tight")
    plt.close()