#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path.cwd()
DEFAULT_OUT_DIR = SCRIPT_DIR / "out"
DEFAULT_SESSION_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
VALID_CONTAINERS = {"auto", "mp4", "mkv"}
VALID_VIDEO_CODECS = {"h264_qsv", "hevc_qsv", "libx264", "copy"}
DURATION_PATTERN = re.compile(r"^([0-9]+|[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?)$")
SEGMENT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*--\s*([0-9]+(?:\.[0-9]+)?):\s*(.+)")


class UserError(Exception):
    pass


@dataclass
class OutputPaths:
    session_dir: Path
    logs_dir: Path
    stills_dir: Path
    asr_dir: Path
    audio_dir: Path
    teacher_output: Path
    screen_output: Path
    teacher_ts: Path
    screen_ts: Path
    teacher_log: Path
    screen_log: Path
    post_log: Path
    denoised_audio: Path


@dataclass
class RecordingProcesses:
    teacher: subprocess.Popen
    screen: subprocess.Popen


@dataclass
class LoggedProcess:
    process: subprocess.Popen
    log_handle: object
    stdout_handle: object | None = None


STOP_REQUESTED = False
CHILD_PROCESSES: list[LoggedProcess] = []


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%F %T')}] {message}")


def die(message: str) -> None:
    raise UserError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record_dual_streams.py",
        description="统一的录制 / 仅 ASR / 仅降噪工具。默认模式为双路录制。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "使用模式：\n"
            "  1) 默认录制模式：同时提供 --teacher-url 和 --screen-url\n"
            "  2) 仅 ASR 模式：提供 --asr-input\n"
            "  3) 仅降噪模式：提供 --denoise-input\n"
            "  4) 仅抽帧模式：提供 --stills-input\n\n"
            "互斥规则：\n"
            "  - 上述四种输入模式互斥\n"
            "  - --record-only 仅在录制模式下有效\n"
            "  - 仅 ASR 模式不需要再传 --enable-asr\n"
        ),
    )

    parser.add_argument("--teacher-url", help="教师流地址（录制模式必填）")
    parser.add_argument("--screen-url", help="屏幕/PPT 流地址（录制模式必填）")
    parser.add_argument("--asr-input", help="对现有本地音视频文件执行 ASR")
    parser.add_argument("--denoise-input", help="对现有本地视频/音频文件做降噪并导出音频")
    parser.add_argument("--stills-input", help="对现有本地视频文件提取场景变化静帧")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出根目录，默认：./out")
    parser.add_argument("--session-name", default=DEFAULT_SESSION_NAME, help="会话目录名，默认：当前时间戳")
    parser.add_argument("--duration", help="录制时长，支持秒数或 HH:MM:SS[.ms]；不传则持续录制，直到手动停止")
    parser.add_argument("--extract-stills", action="store_true", help="录制完成后从 screen 视频抽取场景变化静帧")
    parser.add_argument("--enable-asr", action="store_true", help="录制完成后对 teacher 录制文件执行 ASR")
    parser.add_argument("--record-only", action="store_true", help="仅录制，不执行 ASR / 抽帧等后处理")
    parser.add_argument("--sherpa-dir", help="sherpa-onnx 可执行文件所在目录")
    parser.add_argument("--asr-model-dir", help="FireRedASR2 模型目录，内含 onnx 和 tokens 文件")
    parser.add_argument("--vad-model", help="silero_vad.onnx 路径；不传则自动搜索")
    parser.add_argument("--vad-threshold", type=float, default=0.3, help="VAD 灵敏度阈值(0-1)，越小越容易在短停顿处切分句子，默认：0.3")
    parser.add_argument("--vad-min-silence", type=float, default=0.5, help="VAD 判定静音的最短时长(秒)，越小切分越碎，默认：0.5")
    parser.add_argument("--teacher-global-quality", default="19", help="teacher 流 QSV global quality，默认：19")
    parser.add_argument("--teacher-preset", default="medium", help="teacher 流 QSV preset，默认：medium")
    parser.add_argument("--teacher-video-codec", default="h264_qsv", choices=sorted(VALID_VIDEO_CODECS), help="teacher 流视频编码器，默认：h264_qsv")
    parser.add_argument("--screen-global-quality", default="25", help="screen 流 QSV global quality，默认：25")
    parser.add_argument("--screen-preset", default="medium", help="screen 流 QSV preset，默认：medium")
    parser.add_argument("--screen-video-codec", default="h264_qsv", choices=sorted(VALID_VIDEO_CODECS), help="screen 流视频编码器，默认：h264_qsv")
    parser.add_argument("--screen-fps", default="8", help="screen 输出帧率，默认：8")
    parser.add_argument("--audio-bitrate", default="80k", help="Opus 码率，例如 80k，默认：80k")
    parser.add_argument("--audio-vbr", default="off", choices=["off", "on", "constrained"], help="Opus VBR 模式，默认：off")
    parser.add_argument("--record-audio-codec", default="libopus", choices=["libopus", "copy"], help="录制模式音频编码器，默认：libopus；排障时可改为 copy")
    parser.add_argument("--audio-channels", help="音频输出声道数，仅允许 1 或 2；默认跟随源但不超过 2")
    parser.add_argument("--scene-threshold", default="0.08", help="抽帧场景阈值，默认：0.08")
    parser.add_argument("--container", default="auto", choices=sorted(VALID_CONTAINERS), help="输出容器：auto/mp4/mkv，默认：auto")
    parser.add_argument("--audio-output-format", default="wav", help="仅降噪模式音频输出格式，默认：wav")
    return parser


def validate_args(args: argparse.Namespace) -> str:
    has_record_inputs = bool(args.teacher_url or args.screen_url)
    has_asr_input = bool(args.asr_input)
    has_denoise_input = bool(args.denoise_input)
    has_stills_input = bool(args.stills_input)

    if sum([has_record_inputs, has_asr_input, has_denoise_input, has_stills_input]) == 0:
        die("请指定一种模式：录制模式需同时传 --teacher-url 和 --screen-url；或传 --asr-input；或传 --denoise-input；或传 --stills-input。")

    if has_stills_input and (has_record_inputs or has_asr_input or has_denoise_input):
        die("--stills-input 不能与录制/ASR/降噪模式同时使用。")

    if has_asr_input and has_denoise_input:
        die("--asr-input 与 --denoise-input 不能同时使用。")

    if has_record_inputs and (has_asr_input or has_denoise_input):
        die("录制输入（--teacher-url/--screen-url）不能与 --asr-input 或 --denoise-input 同时使用。")

    if has_record_inputs:
        if not args.teacher_url or not args.screen_url:
            die("录制模式下必须同时提供 --teacher-url 和 --screen-url。")
        mode = "record"
    elif has_asr_input:
        mode = "asr"
    elif has_denoise_input:
        mode = "denoise"
    else:
        mode = "stills"

    if args.record_only and mode != "record":
        die("--record-only 仅能在录制模式下使用。")

    if mode == "record" and args.record_audio_codec == "copy" and args.enable_asr:
        die("录制模式下使用 --record-audio-codec copy 时，不能同时启用 --enable-asr，因为 ASR 需要处理后的 teacher 音频。")

    if args.duration and not DURATION_PATTERN.match(args.duration):
        die("--duration 必须是秒数或 HH:MM:SS[.ms] 格式。")

    if args.audio_channels is not None and args.audio_channels not in {"1", "2"}:
        die("--audio-channels 只能是 1 或 2。")

    input_path = args.asr_input or args.denoise_input or args.stills_input
    if input_path and not Path(input_path).is_file():
        die(f"输入文件不存在：{input_path}")

    if mode in {"record", "asr"} and (args.enable_asr or mode == "asr"):
        validate_asr_requirements(args)

    return mode


def validate_asr_requirements(args: argparse.Namespace) -> None:
    if not args.sherpa_dir:
        die("需要指定 --sherpa-dir。")
    if not args.asr_model_dir:
        die("需要指定 --asr-model-dir。")
    if not Path(args.sherpa_dir).is_dir():
        die(f"Sherpa 目录不存在：{args.sherpa_dir}")
    if not Path(args.asr_model_dir).is_dir():
        die(f"ASR 模型目录不存在：{args.asr_model_dir}")
    if args.vad_model and not Path(args.vad_model).is_file():
        die(f"VAD 模型不存在：{args.vad_model}")


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        die(f"缺少必要命令：{name}")


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout or "") + (result.stderr or "")


def ffmpeg_supports_encoder(name: str) -> bool:
    output = command_output(["ffmpeg", "-hide_banner", "-encoders"])
    return re.search(rf"^\s*[VAS.][A-Z.]*\s+{re.escape(name)}\s", output, re.MULTILINE) is not None


def ffmpeg_supports_muxer(name: str) -> bool:
    return re.search(rf"^[\sE.]+\s+{re.escape(name)}\s", command_output(["ffmpeg", "-hide_banner", "-muxers"]), re.MULTILINE) is not None


def ffmpeg_supports_format(name: str) -> bool:
    return re.search(rf"^[\sDE.]*E\s+{re.escape(name)}\s", command_output(["ffmpeg", "-hide_banner", "-formats"]), re.MULTILINE) is not None


def resolve_container(container: str) -> str:
    if container != "auto":
        return container
    if ffmpeg_supports_muxer("mp4"):
        return "mp4"
    if ffmpeg_supports_muxer("matroska") or ffmpeg_supports_format("matroska"):
        return "mkv"
    die("ffmpeg 既不支持 mp4 也不支持 matroska 封装。")


def build_output_paths(args: argparse.Namespace, container: str) -> OutputPaths:
    out_dir = Path(args.out_dir)
    session_dir = out_dir / args.session_name
    logs_dir = session_dir / "logs"
    stills_dir = session_dir / "screen_stills"
    asr_dir = session_dir / "asr"
    audio_dir = session_dir / "audio"
    logs_dir.mkdir(parents=True, exist_ok=True)

    teacher_output = session_dir / f"teacher.{container}"
    screen_output = session_dir / f"screen.{container}"
    teacher_ts = session_dir / "teacher_recording.ts"
    screen_ts = session_dir / "screen_recording.ts"
    denoised_audio = audio_dir / f"denoised.{args.audio_output_format}"

    return OutputPaths(
        session_dir=session_dir,
        logs_dir=logs_dir,
        stills_dir=stills_dir,
        asr_dir=asr_dir,
        audio_dir=audio_dir,
        teacher_output=teacher_output,
        screen_output=screen_output,
        teacher_ts=teacher_ts,
        screen_ts=screen_ts,
        teacher_log=logs_dir / "teacher_ffmpeg.log",
        screen_log=logs_dir / "screen_ffmpeg.log",
        post_log=logs_dir / "postprocess.log",
        denoised_audio=denoised_audio,
    )


def build_audio_filter() -> str:
    return "highpass=f=80,lowpass=f=9000,afftdn=nf=-20,dynaudnorm=f=150:g=7"


def build_audio_encode_args(args: argparse.Namespace, *, for_recording: bool = True) -> list[str]:
    if not for_recording:
        options = ["-af", build_audio_filter()]
        if args.audio_channels:
            options.extend(["-ac", args.audio_channels])
        return options

    if args.record_audio_codec == "copy":
        return ["-c:a", "copy"]

    options = [
        "-af", build_audio_filter(),
        "-c:a", "libopus",
        "-application", "voip",
        "-b:a", args.audio_bitrate,
        "-vbr", args.audio_vbr,
        "-ar", "48000",
    ]
    if args.audio_channels:
        options.extend(["-ac", args.audio_channels])
    return options


def build_video_encode_args(codec: str, quality: str, preset: str) -> list[str]:
    if codec == "copy":
        return ["-c:v", "copy"]
    if codec in {"h264_qsv", "hevc_qsv"}:
        return [
            "-c:v", codec,
            "-q:v", str(quality),
            "-preset", preset,
        ]
    if codec == "libx264":
        return ["-c:v", codec, "-preset", preset]
    die(f"不支持的视频编码器：{codec}")


def build_duration_args(duration: Optional[str]) -> list[str]:
    return ["-t", duration] if duration else []


def common_movflags(container: str) -> list[str]:
    if container == "mp4":
        return ["-movflags", "+faststart"]
    return []


def run_subprocess_logged(command: list[str], log_file: Path, *, background: bool = False, stdout_path: Optional[Path] = None, check_errors: bool = True, stream_output: bool = False) -> LoggedProcess | None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "a", encoding="utf-8")
    stdout_handle = open(stdout_path, "w", encoding="utf-8") if stdout_path else None
    stdout_target = stdout_handle or log_handle
    stderr_target = sys.stderr if stream_output else log_handle

    if background:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
        )
        logged_process = LoggedProcess(process=process, log_handle=log_handle, stdout_handle=stdout_handle)
        CHILD_PROCESSES.append(logged_process)
        return logged_process

    try:
        result = subprocess.run(command, stdout=stdout_target, stderr=stderr_target, text=True, check=False)
    finally:
        if stdout_handle is not None:
            stdout_handle.close()
        log_handle.close()

    if check_errors and result.returncode != 0:
        die(f"命令执行失败，详见日志：{log_file}")
    return None


def close_logged_process(logged_process: LoggedProcess) -> None:
    if logged_process.stdout_handle is not None and not logged_process.stdout_handle.closed:
        logged_process.stdout_handle.close()
    if not logged_process.log_handle.closed:
        logged_process.log_handle.close()


def stop_children() -> None:
    for logged_process in CHILD_PROCESSES:
        if logged_process.process.poll() is None:
            try:
                logged_process.process.stdin.write("q\n")
                logged_process.process.stdin.flush()
            except (BrokenPipeError, OSError, AttributeError):
                logged_process.process.send_signal(signal.SIGINT)


def finalize_children() -> None:
    import time
    for logged_process in CHILD_PROCESSES:
        proc = logged_process.process
        if proc.poll() is None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                log(f"子进程 {proc.pid} 未响应退出指令，强制终止")
                proc.kill()
                proc.wait()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        close_logged_process(logged_process)


def handle_signal(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    if STOP_REQUESTED:
        return
    STOP_REQUESTED = True
    log("收到停止信号，正在转发给 ffmpeg 子进程...")
    stop_children()


def build_input_args(url: str) -> list[str]:
    if url.startswith("rtsp://"):
        return ["-rtsp_transport", "tcp", "-i", url]
    return ["-rw_timeout", "15000000", "-i", url]


def record_teacher(args: argparse.Namespace, paths: OutputPaths, container: str) -> subprocess.Popen:
    log(f"开始录制 teacher -> {paths.teacher_ts.name} (后台缓冲中)")
    command = [
        "ffmpeg", "-hide_banner", "-y",
        *build_input_args(args.teacher_url),
        *build_duration_args(args.duration),
        "-map", "0:v:0", "-map", "0:a?",
        *build_video_encode_args(args.teacher_video_codec, args.teacher_global_quality, args.teacher_preset),
        "-fps_mode", "passthrough",
        *build_audio_encode_args(args),
        str(paths.teacher_ts),
    ]
    logged_process = run_subprocess_logged(command, paths.teacher_log, background=True)
    assert logged_process is not None
    return logged_process.process


def record_screen(args: argparse.Namespace, paths: OutputPaths, container: str) -> subprocess.Popen:
    log(f"开始录制 screen -> {paths.screen_ts.name} (后台缓冲中)")
    command = [
        "ffmpeg", "-hide_banner", "-y",
        *build_input_args(args.screen_url),
        *build_duration_args(args.duration),
        "-map", "0:v:0", "-map", "0:a?",
    ]
    if args.screen_video_codec != "copy":
        command.extend(["-vf", f"fps={args.screen_fps}"])
    command.extend(build_video_encode_args(args.screen_video_codec, args.screen_global_quality, args.screen_preset))
    if args.screen_video_codec != "copy":
        command.extend(["-r", str(args.screen_fps)])
    command.extend([
        *build_audio_encode_args(args),
        str(paths.screen_ts),
    ])
    logged_process = run_subprocess_logged(command, paths.screen_log, background=True)
    assert logged_process is not None
    return logged_process.process


def wait_recordings(processes: RecordingProcesses, paths: OutputPaths) -> None:
    import time
    while not STOP_REQUESTED:
        t_poll = processes.teacher.poll()
        s_poll = processes.screen.poll()

        if t_poll is not None:
            if t_poll != 0 and not STOP_REQUESTED:
                die(f"Teacher 录制失败，请查看日志：{paths.teacher_log}")
        if s_poll is not None:
            if s_poll != 0 and not STOP_REQUESTED:
                die(f"Screen 录制失败，请查看日志：{paths.screen_log}")

        if t_poll is not None and s_poll is not None:
            return

        time.sleep(0.5)


def remux_ts_to_final(ts_path: Path, final_path: Path, log_path: Path, container: str) -> None:
    if not ts_path.is_file() or ts_path.stat().st_size < 1024:
        die(f"录制临时文件无效或为空，无法转封装：{ts_path}")

    log(f"正在转封装生成最终文件并修复进度条：{ts_path.name} -> {final_path.name}")
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(ts_path),
        "-c", "copy",
        *common_movflags(container),
        str(final_path)
    ]
    run_subprocess_logged(command, log_path)

    ts_path.unlink()
    log(f"转封装完成，已清理临时文件。")


def extract_stills(video_path: Path, args: argparse.Namespace, paths: OutputPaths) -> None:
    if not video_path.is_file():
        die(f"未找到用于抽帧的视频文件：{video_path}")
    paths.stills_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.post_log, "a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().strftime('%F %T')}] 提取场景静帧 -> {paths.stills_dir}\n")

    scene_command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video_path),
        "-vf", f"select='gt(scene,{args.scene_threshold})',showinfo",
        "-fps_mode", "vfr",
        "-strict", "-2",
        "-q:v", "2",
        str(paths.stills_dir / "still_%010d.jpg"),
    ]
    run_subprocess_logged(scene_command, paths.post_log, check_errors=False)

    stills_count = len(list(paths.stills_dir.glob("*.jpg")))
    if stills_count == 0:
        log("提示：未检测到场景变化，未抽取任何静帧。")
    else:
        log(f"静帧提取完成，共抽取出 {stills_count} 张图片。")


def find_sherpa_binary(sherpa_dir: Path) -> Path:
    candidates = [
        sherpa_dir / "sherpa-onnx-vad-with-offline-asr",
        sherpa_dir / "bin" / "sherpa-onnx-vad-with-offline-asr",
        sherpa_dir / "build" / "bin" / "sherpa-onnx-vad-with-offline-asr",
        sherpa_dir / "sherpa-onnx-offline",
        sherpa_dir / "bin" / "sherpa-onnx-offline",
        sherpa_dir / "build" / "bin" / "sherpa-onnx-offline",
        sherpa_dir / "sherpa-onnx",
        sherpa_dir / "bin" / "sherpa-onnx",
        sherpa_dir / "build" / "bin" / "sherpa-onnx",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    die(f"在 {sherpa_dir} 下未找到 sherpa-onnx-vad-with-offline-asr 可执行文件。")


def find_model_file(pattern: str, search_dir: Path) -> Path:
    matches = sorted(search_dir.glob(pattern))
    if not matches:
        die(f"在 {search_dir} 下未找到匹配文件：{pattern}")
    return matches[0]


def find_vad_model(asr_model_dir: Path, vad_model: Optional[str]) -> Path:
    search_root = asr_model_dir.parent
    parent_root = search_root.parent
    grandparent_root = parent_root.parent
    candidates = [
        Path(vad_model) if vad_model else None,
        asr_model_dir / "silero_vad.onnx",
        search_root / "silero_vad.onnx",
        parent_root / "silero_vad.onnx",
        grandparent_root / "silero_vad.onnx",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    die(
        "未找到 silero_vad.onnx。请通过 --vad-model 指定，"
        f"或放到以下位置之一：{asr_model_dir}、{search_root}、{parent_root}、{grandparent_root}"
    )


def extract_teacher_wav(media_path: Path, paths: OutputPaths) -> Path:
    paths.asr_dir.mkdir(parents=True, exist_ok=True)
    wav_file = paths.asr_dir / "teacher_16k_mono.wav"
    with open(paths.post_log, "a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().strftime('%F %T')}] 导出 16k 单声道 WAV -> {wav_file}\n")
    run_subprocess_logged([
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(media_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_file),
    ], paths.post_log)
    return wav_file


def parse_segment_lines(raw_file: Path) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    with open(raw_file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = SEGMENT_PATTERN.search(line)
            if match:
                segments.append((float(match.group(1)), float(match.group(2)), match.group(3).strip()))
    return segments


def format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms_total = round(seconds * 1000)
    hours, rem = divmod(ms_total, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: list[tuple[float, float, str]], srt_file: Path) -> None:
    with open(srt_file, "w", encoding="utf-8") as handle:
        for index, (start, end, text) in enumerate(segments, start=1):
            handle.write(f"{index}\n")
            handle.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            handle.write(f"{text}\n\n")


def run_asr_from_media(media_path: Path, args: argparse.Namespace, paths: OutputPaths) -> None:
    sherpa_bin = find_sherpa_binary(Path(args.sherpa_dir))
    tokens_file = find_model_file("tokens*.txt", Path(args.asr_model_dir))
    vad_model = find_vad_model(Path(args.asr_model_dir), args.vad_model)
    wav_file = extract_teacher_wav(media_path, paths)
    raw_out = paths.asr_dir / "teacher.raw.txt"
    txt_out = paths.asr_dir / "teacher.txt"
    srt_out = paths.asr_dir / "teacher.srt"

    encoder_file = find_model_file("encoder*.onnx", Path(args.asr_model_dir))
    decoder_file = find_model_file("decoder*.onnx", Path(args.asr_model_dir))

    optimal_threads = min(os.cpu_count() or 4, 12)

    asr_args = []
    if encoder_file and decoder_file:
        log(f"运行 FireRedASR2 Transducer + VAD (使用 {optimal_threads} 线程)...")
        with open(paths.post_log, "a", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now().strftime('%F %T')}] 运行 FireRedASR2 Transducer + VAD -> {paths.asr_dir}\n")
        asr_args = [
            f"--fire-red-asr-encoder={encoder_file}",
            f"--fire-red-asr-decoder={decoder_file}",
        ]
    else:
        model_file = find_model_file("model*.onnx", Path(args.asr_model_dir))
        log(f"运行 FireRedASR2 CTC + VAD (使用 {optimal_threads} 线程)...")
        with open(paths.post_log, "a", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now().strftime('%F %T')}] 运行 FireRedASR2 CTC + VAD -> {paths.asr_dir}\n")
        asr_args = [f"--fire-red-asr-ctc={model_file}"]

    run_subprocess_logged([
        str(sherpa_bin),
        f"--silero-vad-model={vad_model}",
        f"--silero-vad-threshold={args.vad_threshold}",
        f"--silero-vad-min-silence-duration={args.vad_min_silence}",
        *asr_args,
        f"--tokens={tokens_file}",
        f"--num-threads={optimal_threads}",
        str(wav_file),
    ], paths.post_log, stdout_path=raw_out, stream_output=True)

    segments = parse_segment_lines(raw_out)
    if not segments:
        die(f"ASR 已执行，但在 {raw_out} 中未找到带时间戳的分段结果。")

    with open(txt_out, "w", encoding="utf-8") as handle:
        for start, end, text in segments:
            handle.write(f"{start} -- {end}: {text}\n")
    write_srt(segments, srt_out)
    log(f"ASR 原始输出：{raw_out}")
    log(f"分段文本：{txt_out}")
    log(f"字幕文件：{srt_out}")


def denoise_media_to_audio(media_path: Path, args: argparse.Namespace, paths: OutputPaths) -> None:
    paths.audio_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(media_path),
        "-vn",
        "-af", build_audio_filter(),
    ]
    if args.audio_channels:
        command.extend(["-ac", args.audio_channels])
    if args.audio_output_format.lower() == "wav":
        command.extend(["-ar", "48000", "-c:a", "pcm_s16le"])
    else:
        command.extend(["-c:a", "libopus", "-b:a", args.audio_bitrate, "-ar", "48000"])
    command.append(str(paths.denoised_audio))
    with open(paths.post_log, "a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().strftime('%F %T')}] 导出降噪音频 -> {paths.denoised_audio}\n")
    run_subprocess_logged(command, paths.post_log)
    log(f"降噪音频已生成：{paths.denoised_audio}")


def ensure_recording_capabilities(args: argparse.Namespace, container: str) -> None:
    for stream_name, codec in (("teacher", args.teacher_video_codec), ("screen", args.screen_video_codec)):
        if codec != "copy" and not ffmpeg_supports_encoder(codec):
            die(f"当前 ffmpeg 不支持 {stream_name} 所需视频编码器：{codec}。")
    if args.record_audio_codec == "libopus" and not ffmpeg_supports_encoder("libopus"):
        die("当前 ffmpeg 不支持 libopus 编码器。")
    if container == "mp4":
        log("已选择 mp4 封装；如果本地 ffmpeg/mp4 不接受 Opus，请改用 --container mkv。")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    mode = validate_args(args)

    require_command("ffmpeg")
    require_command("ffprobe")

    container = resolve_container(args.container)
    paths = build_output_paths(args, container)

    if mode == "record":
        ensure_recording_capabilities(args, container)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if mode == "record":
            processes = RecordingProcesses(
                teacher=record_teacher(args, paths, container),
                screen=record_screen(args, paths, container),
            )
            wait_recordings(processes, paths)

            remux_ts_to_final(paths.teacher_ts, paths.teacher_output, paths.post_log, container)
            remux_ts_to_final(paths.screen_ts, paths.screen_output, paths.post_log, container)

            if not args.record_only:
                if args.extract_stills:
                    extract_stills(paths.screen_output, args, paths)
                if args.enable_asr:
                    run_asr_from_media(paths.teacher_output, args, paths)
        elif mode == "asr":
            run_asr_from_media(Path(args.asr_input), args, paths)
        elif mode == "stills":
            extract_stills(Path(args.stills_input), args, paths)
        else:
            denoise_media_to_audio(Path(args.denoise_input), args, paths)
    finally:
        stop_children()
        finalize_children()

    log(f"完成，输出目录：{paths.session_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
