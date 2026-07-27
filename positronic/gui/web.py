import asyncio
import fractions
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import av
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import pimm
from positronic import geom, utils
from positronic.drivers.roboarm import command
from positronic.policy.harness import Directive


class _JogBody(BaseModel):
    axis: Literal['x', 'y', 'z', 'rx', 'ry', 'rz']
    sign: Literal[-1, 1]
    scale: Literal['fine', 'coarse']
    arm: str | None = None


class _GripBody(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    arm: str | None = None


class _OfferBody(BaseModel):
    sdp: str
    type: Literal['offer']


_TRANSLATION_AXES = {'x': 0, 'y': 1, 'z': 2}
_ROTATION_AXES = {'rx': 0, 'ry': 1, 'rz': 2}

_VIDEO_CLOCK = 90000


def _pkg_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


def _shared_static() -> str:
    return str(Path(__file__).resolve().parent.parent / 'server' / 'static')


def _even(value: int) -> int:
    return max(2, value - value % 2)


def _resize_to_width(rgb: np.ndarray, width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    height = _even(round(h * width / w))
    if (h, w) == (height, width):
        return rgb
    return (
        av.VideoFrame.from_ndarray(rgb, format='rgb24').reformat(width=width, height=height).to_ndarray(format='rgb24')
    )


def _tile(frames: list[np.ndarray], width: int) -> np.ndarray:
    """Stack frames vertically at a common (even) width, so the column encodes as one video stream."""
    width = _even(width)
    return np.concatenate([_resize_to_width(frame, width) for frame in frames], axis=0)


class _LatestFrame:
    """The most recent tiled frame, shared from the control-loop thread to the server's WebRTC encoder tasks.

    ``publish`` stores the frame before bumping ``seq`` so a consumer that observes the new sequence number
    always reads the new frame; consumers poll the counter at 2 ms rather than parking on cross-thread wakeups.
    """

    def __init__(self):
        self.array: np.ndarray | None = None
        self.seq = 0

    def publish(self, array: np.ndarray) -> None:
        self.array = array
        self.seq += 1

    async def next_frame(self, last_seq: int) -> tuple[int, np.ndarray]:
        while self.seq == last_seq:
            await asyncio.sleep(0.002)
        return self.seq, self.array


class _TileTrack(VideoStreamTrack):
    """Sends each freshly published tile as a WebRTC video frame (each subscriber gets its own encoder).

    Event-driven rather than paced: a frame goes out the moment the console publishes it, so the stream
    rate follows the camera rate and no frame sits out a pacing tick.
    """

    def __init__(self, latest: _LatestFrame):
        super().__init__()
        self._latest = latest
        self._seq = 0
        self._epoch: float | None = None

    async def recv(self) -> av.VideoFrame:
        self._seq, rgb = await self._latest.next_frame(self._seq)
        now = time.monotonic()
        if self._epoch is None:
            self._epoch = now
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        frame.pts = int((now - self._epoch) * _VIDEO_CLOCK)
        frame.time_base = fractions.Fraction(1, _VIDEO_CLOCK)
        return frame


class WebEvalUI(pimm.ControlSystem):
    """Headless web operator surface for attended evals.

    Tiles the live eval cameras into a single video stream served to a browser over WebRTC and turns
    Start/Finish/Abort presses into harness directives. A drop-in directive source replacing the
    dearpygui/keyboard drivers, reachable directly on the host IP.

    ``arms`` names the per-arm command-channel suffixes of a multi-arm embodiment (``robot_command.{arm}``,
    ``target_grip.{arm}``); the console then shows an arm selector and jog/grip drive the selected arm. Empty
    ``arms`` targets the bare single-arm channels.
    """

    def __init__(
        self,
        task: str | None = None,
        arms: Sequence[str] = (),
        port=8080,
        fps=20,
        width=640,
        translation_fine=0.01,
        translation_coarse=0.05,
        rotation_fine=2.0,
        rotation_coarse=10.0,
    ):
        self.task = task
        self.arms = tuple(arms)
        self.port = port
        self.fps = fps
        self.width = width
        self.translation_fine = translation_fine
        self.translation_coarse = translation_coarse
        self.rotation_fine = rotation_fine
        self.rotation_coarse = rotation_coarse
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.directive = pimm.ControlSystemEmitter(self)
        self.manual_command = pimm.ControlSystemEmitter(self)

    def _channel(self, base: str, arm: str | None) -> str:
        """The harness command channel for a manual command: per-arm suffixed when the embodiment has arms."""
        if not self.arms:
            return base
        if arm not in self.arms:
            raise HTTPException(status_code=422, detail=f'arm must be one of {list(self.arms)}')
        return f'{base}.{arm}'

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        templates = Jinja2Templates(directory=_pkg_path('templates'))
        names = list(self.cameras)
        latest = _LatestFrame()
        frames: dict[str, np.ndarray] = {}
        pcs: set[RTCPeerConnection] = set()

        app = FastAPI()
        app.mount('/static', StaticFiles(directory=_shared_static()), name='static')
        app.mount('/assets', StaticFiles(directory=_pkg_path('static')), name='assets')

        @app.get('/', response_class=HTMLResponse)
        async def index(request: Request):
            return templates.TemplateResponse(request, 'eval_console.html', {'arms': list(self.arms)})

        @app.post('/webrtc')
        async def webrtc(body: _OfferBody):
            pc = RTCPeerConnection()
            pcs.add(pc)

            @pc.on('connectionstatechange')
            async def on_state_change():
                if pc.connectionState in ('failed', 'closed'):
                    await pc.close()
                    pcs.discard(pc)

            pc.addTrack(_TileTrack(latest))
            await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
            await pc.setLocalDescription(await pc.createAnswer())
            return {'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}

        @app.on_event('shutdown')
        async def shutdown():
            await asyncio.gather(*(pc.close() for pc in pcs), return_exceptions=True)
            pcs.clear()

        @app.post('/directive/{action}')
        async def directive(action: str):
            match action:
                case 'start':
                    self.directive.emit(Directive.RUN(task=self.task), clock.now_ns())
                case 'finish':
                    self.directive.emit(Directive.FINISH(), clock.now_ns())
                case 'abort':
                    self.directive.emit(Directive.ABORT(), clock.now_ns())
                case _:
                    raise HTTPException(status_code=404)

        @app.post('/jog')
        async def jog(body: _JogBody):
            if body.axis in _TRANSLATION_AXES:
                step = self.translation_fine if body.scale == 'fine' else self.translation_coarse
                translation = np.zeros(3)
                translation[_TRANSLATION_AXES[body.axis]] = body.sign * step
                delta = geom.Transform3D(translation=translation)
            elif body.axis in _ROTATION_AXES:
                angle = self.rotation_fine if body.scale == 'fine' else self.rotation_coarse
                rotvec = np.zeros(3)
                rotvec[_ROTATION_AXES[body.axis]] = np.deg2rad(body.sign * angle)
                delta = geom.Transform3D(rotation=geom.Rotation.from_rotvec(rotvec))
            channel = self._channel('robot_command', body.arm)
            self.manual_command.emit({channel: command.CartesianDelta(delta)}, clock.now_ns())

        @app.post('/grip')
        async def grip(body: _GripBody):
            self.manual_command.emit({self._channel('target_grip', body.arm): body.value}, clock.now_ns())

        config = uvicorn.Config(app, host='0.0.0.0', port=self.port)
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        host = utils.resolve_host_ip()
        banner = '=' * 80
        print(banner)
        print(f' >>> WEB eval console available at: http://{host}:{self.port}/ <<<')
        print(banner)

        try:
            # Poll the camera channels much faster than the frame rate so a frame never sits out a
            # sampling tick; ``fps`` only caps how often the (CPU-priced) tile + encode fires. A frame
            # arriving while the cap is closed stays pending and goes out the moment it reopens —
            # otherwise it would be dropped and the stream would sit stale until the next camera frame.
            last_emit = 0.0
            pending = False
            stamps: dict[str, float] = {}
            last_age_log = 0.0
            while not should_stop.value:
                for name in names:
                    cam_msg = self.cameras[name].read()
                    if cam_msg.data is not None and cam_msg.updated:
                        frames[name] = cam_msg.data.array
                        stamps[name] = cam_msg.ts
                        pending = True
                now = time.monotonic()
                if pending and len(frames) == len(names) and now - last_emit >= 1 / self.fps:
                    latest.publish(_tile([frames[name] for name in names], self.width))
                    last_emit = now
                    pending = False
                    if now - last_age_log >= 5.0:
                        # Camera drivers stamp frames with the sensor's epoch-based capture time, so
                        # wall-clock minus stamp is the capture-to-publish latency (pre-WebRTC).
                        wall = time.time()
                        ages = ', '.join(f'{name}={(wall - stamps[name]) * 1000:.0f}ms' for name in names)
                        print(f'Frame age at publish: {ages}', flush=True)
                        last_age_log = now
                if not server_thread.is_alive():
                    raise RuntimeError('Web eval server thread died')
                yield pimm.Sleep(0.005)
        finally:
            server.should_exit = True
            server_thread.join()
