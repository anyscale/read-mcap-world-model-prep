"""Training-sample prep for robot world models, from MCAP recordings, with Ray Data.

One pipeline turns a fleet's MCAP episodes into labeled training clips:

    manifest of episodes (paths + task text, from your catalog)
      -> read_mcap: 3.2 s windows of every camera and joint topic, video
         decoded in the read task at 10 fps and 256x256, keyframe lead-in
         handled, exact resume on row_id
      -> build_samples (fused into the read tasks): a common 100 ms clock,
         actions and joint state resampled onto it, quality checks, one small
         mp4 per camera
      -> captioning: a VLM labels each clip on GPUs (or a stub on a laptop)
      -> WebDataset shards, one sample per clip

Usage:
    python e2e_vlm.py make-demo demo/             # synthetic episodes
    python e2e_vlm.py run demo/manifest.jsonl out/ --checkpoint ckpt/
    python e2e_vlm.py run ... --captioner vlm --model Qwen/Qwen2.5-VL-7B-Instruct

Needs Ray with the read_mcap stack (ray-project/ray#66654, #66655, #66670),
plus: mcap, mcap-ros2-support, av, pillow, aiohttp. The VLM captioner also
needs ``ray[llm]`` and GPUs.
"""

import argparse
import dataclasses
import functools
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -- configuration ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SampleConfig:
    """What a training sample holds and how it is laid out in time."""

    cameras: Dict[str, str]  # sample name -> MCAP topic
    state_topic: str
    action_topic: str
    fps: int = 10  # one slot per frame; must divide one second in ns
    clip_s: float = 3.2  # whole number of slots
    size: Tuple[int, int] = (256, 256)  # (height, width) of decoded frames
    action_hz: int = 50  # commands kept per second: fps must divide it

    @property
    def period_ns(self) -> int:
        return 1_000_000_000 // self.fps

    def validate(self) -> None:
        if 1_000_000_000 % self.fps:
            raise ValueError(f"fps={self.fps} does not divide one second in ns")
        if round(self.clip_s * 1e9) % self.period_ns:
            raise ValueError(f"clip_s={self.clip_s} is not a whole number of slots")
        if self.action_hz % self.fps:
            raise ValueError(f"action_hz={self.action_hz} is not a multiple of fps")


DEMO = SampleConfig(
    cameras={"front": "/camera/front/video", "wrist": "/camera/wrist/video"},
    state_topic="/robot/joint_states",
    action_topic="/robot/joint_commands",
)

# -- the pipeline ---------------------------------------------------------------


def read_clips(paths: List[str], cfg: SampleConfig):
    """Every camera and joint topic of each episode, in windows of ``clip_s``.

    ``anchor="epoch"`` puts window starts on the 100 ms grid the fps thinner
    uses, so every window holds exactly ``clip_s * fps`` slots.
    """
    import ray
    from ray.data.datasource import VideoOptions, WindowSpec

    return ray.data.read_mcap(
        paths,
        topics=[*cfg.cameras.values(), cfg.state_topic, cfg.action_topic],
        read_granularity="window",
        window=WindowSpec(length_s=cfg.clip_s, anchor="epoch", drop_partial=True),
        video=VideoOptions(fps=cfg.fps, resize=cfg.size),
        include_row_id=True,
    )


def run(args: argparse.Namespace) -> None:
    """Read clips, build samples, caption them and write shards."""
    cfg = DEMO
    cfg.validate()
    manifest = [
        json.loads(line) for line in Path(args.manifest).read_text().splitlines()
    ]
    tasks = {m["path"]: m["task"] for m in manifest}
    output = absolute(args.output)
    if args.checkpoint:
        enable_checkpoints(args.checkpoint)

    before = count_samples(output)
    start = time.time()
    clips = read_clips([m["path"] for m in manifest], cfg)
    # Ray Data fuses this CPU step into the read tasks, so decoded frames never
    # leave the task that decoded them: only small mp4s move on. batch_size=None
    # takes each block as the read made it.
    samples = clips.map_batches(
        build_samples,
        batch_size=None,
        batch_format="numpy",
        fn_kwargs={"cfg": cfg, "crash_on": args.crash_on},
    )
    samples = add_captions(samples, args, cfg, tasks)
    samples.write_webdataset(output, encoder=None, min_rows_per_file=args.per_shard)
    print_summary(output, before, time.time() - start)


def enable_checkpoints(path: str) -> None:
    """Record the row_id of every written sample, so a rerun skips those clips."""
    import ray
    from ray.data.checkpoint import CheckpointConfig

    ray.data.DataContext.get_current().checkpoint_config = CheckpointConfig(
        id_column="row_id", checkpoint_path=absolute(path)
    )


def add_captions(samples, args: argparse.Namespace, cfg: SampleConfig, tasks):
    """A caption in every sample's json: the VLM on GPUs, or a CPU stub."""
    if args.captioner == "vlm":
        return vlm_captions(samples, cfg, tasks, args.model, args.gpus)
    # A separate stage, like the GPU captioner. Without one, the write's
    # min_rows_per_file pulls build_samples out of the read tasks and gathers
    # the decoded windows into a few write tasks.
    return samples.map_batches(
        StubCaptioner,
        fn_constructor_kwargs={"tasks": tasks},
        batch_format="numpy",
        concurrency=1,
        num_cpus=0.5,  # resources unlike the read's keep the stages apart
    )


# -- one window -> one sample ---------------------------------------------------


def build_samples(
    batch: Dict[str, np.ndarray], cfg: SampleConfig, crash_on: Optional[str] = None
) -> Dict[str, list]:
    """Turn each decoded window row into one training sample."""
    rows = [{k: v[i] for k, v in batch.items()} for i in range(len(batch["row_id"]))]
    if crash_on and any(crash_on in row["path"] for row in rows):
        time.sleep(10)  # let the other episodes finish and record their clips
        raise RuntimeError(f"simulated crash while processing {crash_on}")
    samples = [build_sample(row, cfg) for row in rows]
    return {key: [s[key] for s in samples] for key in samples[0]} if samples else {}


def build_sample(row: dict, cfg: SampleConfig) -> dict:
    """Frames, actions and state of one window on one 100 ms clock."""
    clock = SlotClock.of_window(row["window_start"], row["window_end"], cfg.period_ns)

    # 1. Each camera's frames into their slots; holes take the previous frame.
    videos, mask = place_frames(row, clock, cfg)

    # 2. Joint messages are still raw bytes: decode them with the row's schemas.
    state_t, state = joint_positions(row, cfg.state_topic)
    action_t, action = joint_positions(row, cfg.action_topic)

    # 3. Onto the clock: state at each slot start, and the command active at
    #    each of the action_hz / fps sub-steps of every slot.
    state_at = interpolate(state_t, state, clock.starts)  # (T, joints)
    actions = hold(
        action_t, action, clock.substeps(cfg.action_hz // cfg.fps)
    )  # (T, K, joints)

    qc = quality_checks(videos, mask, state, action_t)
    return {
        "__key__": hashlib.sha1(row["row_id"].encode()).hexdigest(),
        **{f"{name}.mp4": encode_mp4(clip, cfg.fps) for name, clip in videos.items()},
        "actions.npy": npy_bytes(actions.astype(np.float32)),
        "state.npy": npy_bytes(state_at.astype(np.float32)),
        "mask.npy": npy_bytes(mask),
        "json": json.dumps(clip_meta(row, cfg, len(clock.starts), qc)).encode(),
        "row_id": row["row_id"],
    }


def place_frames(row: dict, clock: "SlotClock", cfg: SampleConfig):
    """Each camera's clip on the clock, and the (T, cameras) mask of real frames."""
    videos, masks = {}, []
    for name, topic in cfg.cameras.items():
        clip, real = clock.place(
            row[f"frames:{topic}"], row[f"frame_times:{topic}"], cfg.size
        )
        videos[name] = clip
        masks.append(real)
    return videos, np.stack(masks, axis=1)


def clip_meta(row: dict, cfg: SampleConfig, slots: int, qc: dict) -> dict:
    """The sample's json: where the clip comes from, its layout and its checks."""
    return {
        "row_id": row["row_id"],
        "path": row["path"],
        "window_start_ns": int(row["window_start"]),
        "window_end_ns": int(row["window_end"]),
        "fps": cfg.fps,
        "slots": slots,
        "action_hz": cfg.action_hz,
        "cameras": list(cfg.cameras),
        "qc": qc,
        "caption": None,
    }


@dataclasses.dataclass(frozen=True)
class SlotClock:
    """The window's slots: slot k covers [starts[k], starts[k] + period)."""

    starts: np.ndarray  # int64 ns
    period_ns: int

    @classmethod
    def of_window(cls, start: int, end: int, period_ns: int) -> "SlotClock":
        first = -(-int(start) // period_ns)  # first slot boundary at or after start
        count = (int(end) - first * period_ns) // period_ns
        return cls(
            np.arange(first, first + count, dtype=np.int64) * period_ns, period_ns
        )

    def place(self, frames, times, size) -> Tuple[np.ndarray, np.ndarray]:
        """One frame per slot, the real flag per slot, holes filled forward."""
        clip = np.zeros((len(self.starts), *size, 3), np.uint8)
        real = np.zeros(len(self.starts), bool)
        slots = (np.asarray(times, np.int64) - self.starts[0]) // self.period_ns
        for slot, frame in zip(slots, frames, strict=True):
            if 0 <= slot < len(clip) and not real[slot]:
                clip[slot], real[slot] = frame, True
        for k in range(1, len(clip)):  # never copies a later frame backwards
            if not real[k] and (real[:k].any()):
                clip[k] = clip[k - 1]
        return clip, real

    def substeps(self, per_slot: int) -> np.ndarray:
        offsets = np.arange(per_slot, dtype=np.int64) * (self.period_ns // per_slot)
        return self.starts[:, None] + offsets[None, :]


def joint_positions(row: dict, topic: str) -> Tuple[np.ndarray, np.ndarray]:
    """Log times and joint positions of one topic's messages in the window."""
    picked = np.flatnonzero(np.asarray(row["topic"]) == topic)
    if len(picked) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 0))
    channel = next(c for c in row["channels"] if c["topic"] == topic)
    decode = decoder_for(channel)
    times = np.asarray(row["log_time"])[picked]
    values = np.array([decode(row["data"][i]).position for i in picked], np.float64)
    return times, values


_DECODERS: Dict[tuple, object] = {}  # one decoder per schema, per worker


def decoder_for(channel: dict):
    """A message decoder built from the schema the window row carries."""
    from mcap.records import Schema
    from mcap_ros2.decoder import DecoderFactory

    key = (
        channel["schema_name"],
        bytes(channel["schema_data"]),
        channel["message_encoding"],
    )
    if key not in _DECODERS:
        schema = Schema(
            id=1, name=key[0], encoding=channel["schema_encoding"], data=key[1]
        )
        _DECODERS[key] = DecoderFactory().decoder_for(key[2], schema)
    return _DECODERS[key]


def interpolate(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Each column of values, linear in time; flat beyond the first and last."""
    if len(times) == 0:
        return np.zeros((len(at), 0))
    return np.stack(
        [np.interp(at, times, values[:, j]) for j in range(values.shape[1])], axis=1
    )


def hold(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The last value at or before each time; the first value before any."""
    if len(times) == 0:
        return np.zeros((*at.shape, 0))
    index = np.clip(np.searchsorted(times, at, side="right") - 1, 0, len(times) - 1)
    return values[index]


def quality_checks(videos, mask, state, action_t) -> dict:
    """Cheap flags a curation step can filter on.

    Brightness and motion are mean 8-bit pixel levels over the real frames.
    """
    report = {}
    for c, (name, clip) in enumerate(videos.items()):
        real = clip[mask[:, c]]
        brightness = float(real.mean()) if len(real) else 0.0
        motion = (
            float(np.abs(np.diff(real.astype(np.int16), axis=0)).mean())
            if len(real) > 1
            else 0.0
        )
        report[name] = {
            "missing_frames": int((~mask[:, c]).sum()),
            "dark": brightness < 12.0,
            "frozen": motion < 0.5,
        }
    report["joints_finite"] = bool(np.isfinite(state).all())
    report["max_action_gap_ms"] = (
        round(float(np.diff(action_t).max()) / 1e6, 1) if len(action_t) > 1 else None
    )
    return report


def encode_mp4(clip: np.ndarray, fps: int) -> bytes:
    """H.264 at the clip's size: one keyframe at frame 0, no B-frames, so
    video frame k is slot k and every clip decodes on its own."""
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = clip.shape[1:3]
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "crf": "23",
            "preset": "veryfast",
            "g": str(len(clip)),
            "bf": "0",
        }
        for frame in clip:
            container.mux(
                stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
            )
        container.mux(stream.encode())
    return buffer.getvalue()


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


# -- captioning -----------------------------------------------------------------

CAPTION_PROMPT = """You label short robot manipulation clips for training data.
Task given to the robot: {task}
The images are {n} frames of the front camera, then {n} frames of the wrist
camera, both evenly spaced over {seconds:.1f} seconds.
Answer in JSON: a one-sentence summary of what the robot does, the action as a
short verb phrase, the objects it touches or moves, whether the task succeeded,
and any quality problems in the images."""

# vLLM constrains decoding to this schema, so every caption parses.
CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "action": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "outcome": {"enum": ["success", "failure", "unclear"]},
        "quality": {
            "type": "array",
            "items": {"enum": ["blurry", "occluded", "dark", "static"]},
        },
    },
    "required": ["summary", "action", "objects", "outcome", "quality"],
}


class StubCaptioner:
    """A CPU stand-in for the VLM, so the pipeline runs on a laptop."""

    def __init__(self, tasks: Dict[str, str]):
        self.tasks = tasks

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        metas = [json.loads(m) for m in batch["json"]]
        for meta in metas:
            flags = [
                f"{camera} {flag}"
                for camera, qc in meta["qc"].items()
                if isinstance(qc, dict)
                for flag in ("dark", "frozen")
                if qc[flag]
            ]
            meta["caption"] = {
                "summary": f"[stub] {self.tasks[meta['path']]}",
                "quality": flags,
            }
        return {
            **batch,
            "json": np.array([json.dumps(m).encode() for m in metas], dtype=object),
        }


def vlm_captions(ds, cfg: SampleConfig, tasks: Dict[str, str], model: str, gpus: int):
    """Caption every clip with a VLM: Ray Data LLM runs vLLM on GPU actors."""
    from ray.data.llm import build_processor, vLLMEngineProcessorConfig

    frames_per_camera = 4
    config = vLLMEngineProcessorConfig(
        model_source=model,
        engine_kwargs={
            "max_model_len": 16384,
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"image": frames_per_camera * len(cfg.cameras)},
        },
        prepare_multimodal_stage=True,
        batch_size=16,
        concurrency=gpus,
    )
    processor = build_processor(
        config,
        preprocess=functools.partial(
            caption_request, cfg=cfg, tasks=tasks, n=frames_per_camera
        ),
        postprocess=attach_caption,
    )
    return processor(ds)


def caption_request(
    row: dict, cfg: SampleConfig, tasks: Dict[str, str], n: int
) -> dict:
    """Chat messages for one clip: the prompt, then n frames per camera."""
    from PIL import Image

    meta = json.loads(row["json"])
    images = [
        {"type": "image_pil", "image_pil": Image.fromarray(frame)}
        for name in cfg.cameras
        for frame in sample_frames(row[f"{name}.mp4"], n)
    ]
    prompt = CAPTION_PROMPT.format(task=tasks[meta["path"]], n=n, seconds=cfg.clip_s)
    return {
        **row,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}, *images]}
        ],
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 200,
            "detokenize": False,
            # A JSON string, so Arrow does not reshape the schema into structs.
            "structured_outputs": {"json": json.dumps(CAPTION_SCHEMA)},
        },
    }


def attach_caption(row: dict) -> dict:
    """Keep the sample's files and put the parsed caption into its json."""
    meta = json.loads(row["json"])
    try:
        meta["caption"] = json.loads(row["generated_text"])
    except (TypeError, ValueError):  # cut off by max_tokens
        meta["caption"] = {"raw": row["generated_text"]}
    files = {
        k: row[k]
        for k in row
        if k.endswith((".mp4", ".npy")) or k in ("__key__", "row_id")
    }
    return {**files, "json": json.dumps(meta).encode()}


def sample_frames(mp4: bytes, n: int) -> List[np.ndarray]:
    """n frames spread evenly over a clip."""
    import av

    with av.open(io.BytesIO(mp4)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    picks = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in picks]


# -- reporting --------------------------------------------------------------------


def absolute(path: str) -> str:
    """A local path made absolute, for Ray workers; a URI such as s3:// as is."""
    return path if "://" in path else str(Path(path).resolve())


def count_samples(output: str) -> int:
    """Samples already in a local output directory; 0 for a bucket."""
    if "://" in output or not Path(output).exists():
        return 0
    total = 0
    for shard in Path(output).glob("*.tar"):
        with tarfile.open(shard) as tar:
            total += sum(1 for m in tar.getmembers() if m.name.endswith(".json"))
    return total


def print_summary(output: str, before: int, seconds: float) -> None:
    """Totals of a local output, and its earliest clip."""
    if "://" in output:
        print(f"\nwrote samples to {output} in {seconds:.1f} s")
        return
    shards = sorted(Path(output).glob("*.tar"))
    total = count_samples(output)
    size = sum(t.stat().st_size for t in shards)
    print(
        f"\nwrote {total - before} new samples in {seconds:.1f} s ({total} total, "
        f"{len(shards)} shards, {size / max(total, 1) / 1e3:.0f} KB per sample)"
    )
    if shards:
        print_first_clip(shards[0])


def print_first_clip(shard: Path) -> None:
    """The files and metadata of the earliest clip in a shard."""
    with tarfile.open(shard) as tar:
        members = tar.getmembers()
        metas = [
            json.loads(tar.extractfile(m).read())
            for m in members
            if m.name.endswith(".json")
        ]
    first = min(metas, key=lambda m: (m["path"], m["window_start_ns"]))
    key = hashlib.sha1(first["row_id"].encode()).hexdigest()
    files = sorted(m.name[len(key) + 1 :] for m in members if m.name.startswith(key))
    print("first clip:", ", ".join(files))
    shown = ("row_id", "window_start_ns", "window_end_ns", "qc", "caption")
    print(json.dumps({k: first[k] for k in shown}, indent=2))


# -- synthetic episodes -------------------------------------------------------------

VIDEO_MSGDEF = """builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

JOINT_STATE_MSGDEF = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

DEMO_TASKS = [
    "push the red block to the left edge of the table",
    "move the arm over the red block and hold still",
    "sweep the red block toward the robot",
]


def make_demo(args: argparse.Namespace) -> None:
    """Synthetic episodes shaped like fleet data: two H.264 cameras at 30 fps
    with a keyframe every 2 s, joint commands at 50 Hz and joint states at
    200 Hz, in ROS 2 messages, plus a manifest of paths and task text."""
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in range(args.episodes):
        path = out / f"episode_{episode:03d}.mcap"
        write_episode(path, seconds=args.seconds, seed=episode)
        rows.append(
            {
                "path": str(path),
                "episode_id": path.stem,
                "task": DEMO_TASKS[episode % 3],
            }
        )
        print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", out / "manifest.jsonl")


@dataclasses.dataclass(frozen=True)
class Event:
    """One message to write: when it was logged, on which topic, and what."""

    log_time: int
    topic: str
    schema: str  # "video" or "joints"
    message: dict
    publish_time: int  # the header stamp: capture or command time


def write_episode(path: Path, seconds: float, seed: int) -> None:
    """One synthetic episode as a ROS 2 MCAP file."""
    rng = np.random.default_rng(seed)
    # One episode per hour, starting off the 3.2 s window grid like a real one.
    t0 = 1_760_000_000_000_000_000 + seed * 3_600_000_000_000 + (seed + 1) * 700_000_000
    joints = ArmMotion(rng)
    write_mcap(
        path, joint_events(joints, t0, seconds) + camera_events(joints, t0, seconds)
    )


def joint_events(joints: "ArmMotion", t0: int, seconds: float) -> List[Event]:
    """Commands at 50 Hz and measured state at 200 Hz, each logged 1 ms late."""
    events = []
    for topic, hz, position in (
        (DEMO.action_topic, 50, joints.target),
        (DEMO.state_topic, 200, joints.state),
    ):
        for i in range(int(seconds * hz)):
            t = t0 + i * (1_000_000_000 // hz)
            message = joint_msg(t, position(t))
            events.append(Event(t + 1_000_000, topic, "joints", message, t))
    return events


def camera_events(joints: "ArmMotion", t0: int, seconds: float) -> List[Event]:
    """Two H.264 cameras at 30 fps, logged 25 and 40 ms after capture."""
    events = []
    for name, delay_ms in (("front", 25), ("wrist", 40)):
        captures = [t0 + round(i * 1e9 / 30) for i in range(int(seconds * 30))]
        frames = [render(name, joints.state(c)) for c in captures]
        packets = h264_packets(frames, fps=30, gop=60)
        for capture, packet in zip(captures, packets, strict=True):
            message = {
                "timestamp": stamp(capture),
                "frame_id": name,
                "data": packet,
                "format": "h264",
            }
            log_time = capture + delay_ms * 1_000_000
            events.append(
                Event(log_time, DEMO.cameras[name], "video", message, capture)
            )
    return events


def write_mcap(path: Path, events: List[Event]) -> None:
    """The events in log-time order, as ROS 2 messages in 4 MiB chunks."""
    from mcap_ros2.writer import Writer

    with open(path, "wb") as stream:
        writer = Writer(stream, chunk_size=4 * 1024 * 1024)
        schemas = {
            "video": writer.register_msgdef(
                "foxglove_msgs/msg/CompressedVideo", VIDEO_MSGDEF
            ),
            "joints": writer.register_msgdef(
                "sensor_msgs/msg/JointState", JOINT_STATE_MSGDEF
            ),
        }
        for sequence, event in enumerate(sorted(events, key=lambda e: e.log_time)):
            writer.write_message(
                topic=event.topic,
                schema=schemas[event.schema],
                message=event.message,
                log_time=event.log_time,
                publish_time=event.publish_time,
                sequence=sequence,
            )
        writer.finish()


class ArmMotion:
    """A two-joint arm: smooth commanded targets, state lagging behind them."""

    def __init__(self, rng: np.random.Generator):
        self.freq = rng.uniform(0.1, 0.4, size=(2, 3))
        self.phase = rng.uniform(0, 2 * np.pi, size=(2, 3))

    def target(self, t: int) -> np.ndarray:
        waves = np.sin(2 * np.pi * self.freq * (t / 1e9) + self.phase)
        return 0.6 * waves.sum(axis=1) / 3 + np.array([0.9, -0.6])

    def state(self, t: int) -> np.ndarray:
        return self.target(t - 80_000_000)  # 80 ms behind the command


def render(
    camera: str, q: np.ndarray, width: int = 320, height: int = 240
) -> np.ndarray:
    """One camera's view of the arm at joint angles q."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (178, 182, 186))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 190, width, height], fill=(120, 96, 72))  # table
    base = np.array([160.0, 190.0])
    elbow = base + 80 * np.array([np.cos(q[0]), -np.sin(q[0])])
    hand = elbow + 65 * np.array([np.cos(q[0] + q[1]), -np.sin(q[0] + q[1])])
    if camera == "wrist":  # follows the hand
        shift = np.array([width / 2, height / 2]) - hand
        base, elbow, hand = base + shift, elbow + shift, hand + shift
    draw.line([*base, *elbow], fill=(40, 70, 140), width=14)
    draw.line([*elbow, *hand], fill=(60, 110, 190), width=11)
    draw.ellipse(
        [hand[0] - 9, hand[1] - 9, hand[0] + 9, hand[1] + 9], fill=(230, 230, 230)
    )
    block = np.array([200.0, 172.0]) + (
        np.array([width / 2, height / 2]) - hand if camera == "wrist" else 0
    )
    draw.rectangle(
        [block[0] - 14, block[1] - 14, block[0] + 14, block[1] + 14], fill=(200, 40, 40)
    )
    return np.asarray(image)


def h264_packets(frames: List[np.ndarray], fps: int, gop: int) -> List[bytes]:
    """One Annex-B access unit per frame, parameter sets on every keyframe and
    no B-frames, as robot recorders commonly write video."""
    import av

    container = av.open(io.BytesIO(), mode="w", format="h264")
    stream = container.add_stream("libx264", rate=fps)
    stream.height, stream.width = frames[0].shape[:2]
    stream.pix_fmt = "yuv420p"
    stream.options = {
        "g": str(gop),
        "bf": "0",
        "preset": "veryfast",
        "crf": "23",
        "x264-params": "repeat-headers=1:scenecut=0",
    }
    packets = []
    for frame in frames:
        packets.extend(
            bytes(p)
            for p in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
        )
    packets.extend(bytes(p) for p in stream.encode())
    return packets


def stamp(t: int) -> dict:
    return {"sec": t // 1_000_000_000, "nanosec": t % 1_000_000_000}


def joint_msg(t: int, position: np.ndarray) -> dict:
    return {
        "header": {"stamp": stamp(t), "frame_id": ""},
        "name": ["shoulder", "elbow"],
        "position": [float(v) for v in position],
        "velocity": [],
        "effort": [],
    }


# -- command line ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "make-demo", help="write synthetic episodes and a manifest"
    )
    demo.add_argument("out_dir")
    demo.add_argument("--episodes", type=int, default=3)
    demo.add_argument("--seconds", type=float, default=20.0)
    go = commands.add_parser(
        "run", help="turn the manifest's episodes into training samples"
    )
    go.add_argument("manifest", help="JSON lines with path and task per episode")
    go.add_argument("output", help="directory or bucket prefix for WebDataset shards")
    go.add_argument(
        "--checkpoint", help="where to record finished clips, for exact resume"
    )
    go.add_argument("--captioner", choices=["stub", "vlm"], default="stub")
    go.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    go.add_argument("--gpus", type=int, default=1)
    go.add_argument("--per-shard", type=int, default=1000, help="samples per shard")
    go.add_argument("--crash-on", help="demo only: fail while processing this episode")
    args = parser.parse_args()
    make_demo(args) if args.command == "make-demo" else run(args)


if __name__ == "__main__":
    main()
