import sys
import numpy as np
from pathlib import Path

# allow scripts to access project modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.video_reader import VideoReader
from core.detector import PersonDetector
from core.optical_flow import OpticalFlowEngine
from core.physics import PhysicsEngine


MPP = 0.02


def precompute(video_path: str):

    video_file = Path(video_path)
    name = video_file.stem

    out = Path("cache") / name
    out.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(str(video_file))
    det = PersonDetector(conf=0.35)
    flow = OpticalFlowEngine()
    phys = PhysicsEngine(fps=reader.fps, meters_per_pixel=MPP)

    density_list = []
    pressure_list = []
    ti_list = []
    re_list = []
    shock_list = []

    fi = 0
    density = np.zeros((32, 32), dtype=np.float32)

    print(f"Processing {video_file} ({reader.total_frames} frames)...")

    while True:
        ret, frame = reader.read_frame()

        if not ret or frame is None:
            break

        H, W = frame.shape[:2]

        # run person detection every 3 frames
        if fi % 3 == 0:
            dets = det.detect(frame)

            density = det.to_density_grid(
                dets,
                H,
                W,
                meters_per_pixel=MPP,
            )

        f = flow.update(frame)

        if f is not None:
            m = phys.compute(density, f)

            density_list.append(m["density_grid"])
            pressure_list.append(m["pressure_norm"])
            ti_list.append(m["TI"])
            re_list.append(m["Re"])
            shock_list.append(m["shockwave_flag"])

        fi += 1

        if fi % 100 == 0:
            print(f"{fi}/{reader.total_frames}")

    reader.release()

    if len(ti_list) == 0:
        print("No frames processed.")
        return

    np.save(out / "density.npy", np.stack(density_list))
    np.save(out / "pressure.npy", np.stack(pressure_list))
    np.save(out / "ti_series.npy", np.array(ti_list))
    np.save(out / "re_series.npy", np.array(re_list))
    np.save(out / "shock_series.npy", np.array(shock_list))

    print(f"Saved {len(ti_list)} frames to {out}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/precompute.py <video_path>")
        sys.exit(1)

    precompute(sys.argv[1])
