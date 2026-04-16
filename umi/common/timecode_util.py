from typing import Union
from fractions import Fraction
import datetime
import os
import av


def timecode_to_seconds(
        timecode: str, frame_rate: Union[int, float, Fraction]
        ) -> Union[float, Fraction]:
    """
    Convert non-skip frame timecode into seconds since midnight
    """
    int_frame_rate = round(frame_rate)
    h, m, s, f = [int(x) for x in timecode.split(':')]
    frames = (3600 * h + 60 * m + s) * int_frame_rate + f
    seconds = frames / frame_rate
    return seconds


def _parse_creation_time(creation_time: str) -> datetime.datetime:
    """
    Parse common QuickTime/ffmpeg creation_time strings.
    Examples:
      2025-12-27T03:21:10.000000Z
      2025-12-27T03:21:10Z
    """
    # normalize
    ct = creation_time.strip()
    if ct.endswith('Z'):
        ct = ct[:-1]
    # try microseconds then no-microseconds
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(ct, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognized creation_time format: {creation_time}")


def stream_get_start_datetime(
    stream: av.stream.Stream,
    container: av.container.InputContainer = None,
    mp4_path: str = None
) -> datetime.datetime:
    """
    Best-effort start datetime for the first frame of a video.

    Priority:
      1) (creation_time + timecode)  [original UMI behavior]
      2) creation_time only          [fallback]
      3) container/stream start_time [fallback]
      4) file mtime                  [fallback, always available]
    """
    md = stream.metadata or {}

    frame_rate = stream.average_rate

    tc = md.get('timecode', None)
    creation_time = md.get('creation_time', None)

    # 1) Original UMI: creation_time + timecode -> high precision
    if (tc is not None) and (creation_time is not None) and (frame_rate is not None):
        seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
        delta = datetime.timedelta(seconds=seconds_since_midnight)

        create_datetime = _parse_creation_time(creation_time)
        create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        return create_datetime + delta

    # 2) creation_time exists but no timecode: use creation_time directly
    if creation_time is not None:
        try:
            return _parse_creation_time(creation_time)
        except Exception:
            pass

    # 3) Use container/stream start_time if available (in AV_TIME_BASE units)
    # container.start_time is in microseconds (AV_TIME_BASE = 1e6)
    if container is not None and container.start_time is not None:
        # This is relative, not absolute wall-clock; still useful for ordering within a session.
        # Use an arbitrary anchor date (epoch) for deterministic behavior.
        anchor = datetime.datetime(1970, 1, 1)
        return anchor + datetime.timedelta(microseconds=int(container.start_time))

    if stream.start_time is not None and stream.time_base is not None:
        # stream.start_time * time_base -> seconds
        anchor = datetime.datetime(1970, 1, 1)
        seconds = float(stream.start_time * stream.time_base)
        return anchor + datetime.timedelta(seconds=seconds)

    # 4) Last resort: file modification time (absolute and stable)
    if mp4_path is not None and os.path.exists(mp4_path):
        ts = os.path.getmtime(mp4_path)
        return datetime.datetime.fromtimestamp(ts)

    # If everything is missing (rare)
    return datetime.datetime(1970, 1, 1)


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream, container=container, mp4_path=mp4_path)
