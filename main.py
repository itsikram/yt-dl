from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import tempfile
import subprocess
from typing import Tuple
import logging
import base64
import threading
import logging
import urllib.parse
import uuid
import asyncio
import time

import yt_dlp
try:
    import imageio_ffmpeg  # provides a bundled ffmpeg binary via pip
except Exception:
    imageio_ffmpeg = None

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global CLEANUP_TASK, COOKIES_FILE
    try:
        available = _has_ffmpeg()
        logger.info("Startup: DOWNLOAD_DIR=%s", DOWNLOAD_DIR)
        logger.info("Startup: FFMPEG_EXE=%s available=%s", FFMPEG_EXE, available)
        cookies_b64 = os.getenv("YTDLP_COOKIES_B64")
        if cookies_b64 and not COOKIES_FILE:
            try:
                decoded = base64.b64decode(cookies_b64)
                import tempfile as _tf
                cookies_tmp = _tf.NamedTemporaryFile(prefix="ytvdl_cookies_", suffix=".txt", delete=False)
                with cookies_tmp as f:
                    f.write(decoded)
                COOKIES_FILE = cookies_tmp.name
                logger.info("Startup: Loaded cookies file from env into %s", COOKIES_FILE)
            except Exception:
                logger.exception("Startup: Failed to decode/write YTDLP_COOKIES_B64")
    except Exception:
        logger.exception("Startup: failed to resolve ffmpeg or log env")
    if CLEANUP_TASK is None or CLEANUP_TASK.done():
        CLEANUP_TASK = asyncio.create_task(_cleanup_downloads_loop())
    try:
        yield
    finally:
        if CLEANUP_TASK is not None:
            CLEANUP_TASK.cancel()
            try:
                await CLEANUP_TASK
            except BaseException:
                pass
            CLEANUP_TASK = None

app = FastAPI(title="YouTube MP4 Downloader", lifespan=_lifespan)

# Logger
logger = logging.getLogger("ytvdl")

# Allow cross-origin video playback from web apps (Android/iOS browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "HEAD"],
    allow_headers=["*"],
)


# Directory to persist downloadable files and expose via /files.
# Default to OS temp dir for compatibility with read-only roots on some hosts.
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR") or os.path.join(tempfile.gettempdir(), "ytvdl_downloads")
try:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
except Exception as e:
    raise RuntimeError(f"Cannot create DOWNLOAD_DIR '{DOWNLOAD_DIR}': {e}")
app.mount("/files", StaticFiles(directory=DOWNLOAD_DIR), name="files")


# Allow overriding ffmpeg path via environment; otherwise resolve automatically
FFMPEG_EXE = os.getenv("FFMPEG_EXE")
# In-memory progress store keyed by a client/job-provided progress_id
JOB_PROGRESS: dict[str, dict] = {}

# Optional cookies file to pass to yt-dlp (Netscape format), populated from env
COOKIES_FILE: str | None = None


def _make_progress_hook(progress_id: str):
    def _hook(d: dict) -> None:
        status = d.get("status")
        if progress_id not in JOB_PROGRESS:
            JOB_PROGRESS[progress_id] = {}
        entry = JOB_PROGRESS[progress_id]
        if status == "downloading":
            downloaded = d.get("downloaded_bytes") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            pct = float(downloaded) / float(total) * 100.0 if total else 0.0
            entry.update({
                "stage": "downloading",
                "status": "running",
                "downloaded_bytes": int(downloaded),
                "total_bytes": int(total) if total else None,
                "pct": round(pct, 2),
                "speed_bps": d.get("speed"),
                "eta_seconds": d.get("eta"),
            })
        elif status == "finished":
            entry.update({
                "stage": "postprocessing",
                "status": "running",
                "pct": 100.0,
            })
    return _hook


# Auto-delete files from the public downloads directory after 5 hours
FILE_TTL_SECONDS = 5 * 60 * 60  # 5 hours
CLEANUP_INTERVAL_SECONDS = 10 * 60  # scan every 10 minutes
CLEANUP_TASK: asyncio.Task | None = None


def _resolve_ffmpeg_exe() -> str | None:
    # Prefer bundled imageio-ffmpeg if available
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    # Fallback to system ffmpeg
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def _has_ffmpeg() -> bool:
    global FFMPEG_EXE
    if FFMPEG_EXE is None:
        FFMPEG_EXE = _resolve_ffmpeg_exe()
    return FFMPEG_EXE is not None


def _delete_path_safely(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    except Exception:
        # Best-effort delete; ignore failures (locked file, etc.)
        pass


def _cleanup_downloads_once(now_ts: float) -> None:
    if not os.path.isdir(DOWNLOAD_DIR):
        return
    try:
        for entry in os.scandir(DOWNLOAD_DIR):
            try:
                # Only target regular files; skip directories created by other tools
                if entry.is_file():
                    mtime = entry.stat().st_mtime
                    if (now_ts - mtime) > FILE_TTL_SECONDS:
                        _delete_path_safely(entry.path)
            except FileNotFoundError:
                # Entry may have been removed concurrently
                continue
            except Exception:
                # Continue on best-effort basis
                continue
    except Exception:
        # If scandir itself fails, skip this round
        pass


async def _cleanup_downloads_loop() -> None:
    # Periodically scan and delete expired files
    while True:
        now_ts = time.time()
        _cleanup_downloads_once(now_ts)
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        except BaseException:
            # Exit silently on cancellation
            break


def _progress_printer(d: dict) -> None:
    status = d.get("status")
    if status == "downloading":
        downloaded = d.get("downloaded_bytes") or 0
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        pct = (downloaded / total * 100.0) if total else 0.0
        speed = d.get("speed")
        eta = d.get("eta")
        total_str = yt_dlp.utils.format_bytes(total) if total else "unknown"
        speed_str = (yt_dlp.utils.format_bytes(speed) + "/s") if speed else "--"
        eta_str = yt_dlp.utils.formatSeconds(eta) if eta else "--"
        print(f"\r[download] {pct:6.2f}% of {total_str} at {speed_str} ETA {eta_str}", end="", flush=True)
    elif status == "finished":
        filename = d.get("filename") or "file"
        print(f"\n[download] Completed: {filename}")


def _download_video_to_tmp(
    url: str,
    progress_id: str | None = None,
    cookies_file: str | None = None,
    target_height: int | None = None,
    fast: bool = True,
) -> Tuple[str, str, str]:
    """
    Download a YouTube video to a temporary directory and return
    (temp_dir, file_path, download_title).

    If FFmpeg is available, allow merging/conversion to MP4. Otherwise, try to
    fetch a progressive MP4.
    """
    temp_dir = tempfile.mkdtemp(prefix="ytvdl_")

    ffmpeg_available = _has_ffmpeg()

    # Prefer MP4 output. With FFmpeg we can merge/convert; without, try progressive MP4.
    if ffmpeg_available:
        ydl_opts = {
            "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,
            "no_warnings": True,
            "restrictfilenames": True,
            # Prefer H.264 video (avc1) + AAC (m4a) and remux to mp4
            "merge_output_format": "mp4",
            "postprocessors": [
                {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
            ],
            # Prioritize H.264/AAC; if a height is requested, try to cap at that height to avoid transcoding
            # Always end with bestvideo+bestaudio/best as ultimate fallback
            "format": (
                f"bestvideo[ext=mp4][vcodec^=avc1][height<={target_height}]+bestaudio[ext=m4a]/"
                f"bestvideo[height<={target_height}]+bestaudio[ext=m4a]/"
                f"bestvideo[height<={target_height}]+bestaudio/"
                f"best[ext=mp4][height<={target_height}]/best[height<={target_height}]/"
                "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo+bestaudio[ext=m4a]/"
                "bestvideo+bestaudio/best[ext=mp4]/best"
            ) if target_height else "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best",
        }
        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file
        if _has_ffmpeg():
            # Point yt-dlp to the bundled/system ffmpeg executable
            ydl_opts["ffmpeg_location"] = FFMPEG_EXE
        if progress_id:
            ydl_opts["progress_hooks"] = [_make_progress_hook(progress_id)]
        else:
            ydl_opts["progress_hooks"] = [_progress_printer]
    else:
        ydl_opts = {
            "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
            "noplaylist": True,
            "quiet": False,
            "no_warnings": True,
            "restrictfilenames": True,
            # Force a single progressive stream (has both audio+video) to avoid ffmpeg merge
            # Prefer MP4; fall back to any progressive format with audio
            # Always end with best as ultimate fallback
            "format": (
                f"best[acodec!=none][vcodec!=none][ext=mp4][height<={target_height}]/"
                f"best[acodec!=none][vcodec!=none][height<={target_height}]/"
                "best[acodec!=none][vcodec!=none][ext=mp4]/"
                "best[acodec!=none][vcodec!=none]/"
                "best"
            ) if target_height else "best[acodec!=none][vcodec!=none][ext=mp4]/best[acodec!=none][vcodec!=none]/best",
        }
        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file
        if progress_id:
            ydl_opts["progress_hooks"] = [_make_progress_hook(progress_id)]
        else:
            ydl_opts["progress_hooks"] = [_progress_printer]

    # Try download with preferred format selector, fallback to simpler selector on error
    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Store prepared filename before exiting context
            prepared = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as format_error:
        # If format selector failed, retry with a more lenient selector
        if "Requested format is not available" in str(format_error) or "format is not available" in str(format_error):
            logger.warning("Format selector failed, retrying with lenient selector for url=%s", url)
            # Create a simpler, more lenient format selector
            if ffmpeg_available:
                ydl_opts["format"] = "bestvideo+bestaudio/best"
            else:
                ydl_opts["format"] = "best[acodec!=none][vcodec!=none]/best"
            # Retry with lenient selector
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    # Store prepared filename before exiting context
                    prepared = ydl.prepare_filename(info)
            except yt_dlp.utils.DownloadError as retry_error:
                # Clean up and re-raise the original error
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.exception("yt-dlp download error (retry failed) for url=%s", url)
                raise HTTPException(status_code=400, detail=f"Download error: {str(retry_error)}")
        else:
            # For other download errors, clean up and re-raise
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.exception("yt-dlp download error for url=%s", url)
            raise HTTPException(status_code=400, detail=f"Download error: {str(format_error)}")
    
    try:
        # Continue with file processing
        # Derive final filepath; postprocessing may change extension
        base, _ = os.path.splitext(prepared)
        candidates = [
            prepared,
            base + ".mp4",
            base + ".mkv",  # sometimes merges default to mkv
            base + ".webm",
        ]
        file_path = next((p for p in candidates if os.path.exists(p)), None)
        if not file_path:
            raise HTTPException(status_code=500, detail="Download succeeded but file not found.")

        # Enforce MP4 output
        if not file_path.lower().endswith(".mp4"):
            if ffmpeg_available:
                # If FFmpeg was available, we expected mp4 already
                raise HTTPException(status_code=500, detail="Failed to remux to MP4.")
            # Without FFmpeg we cannot convert; instruct user
            raise HTTPException(
                status_code=415,
                detail="MP4 not available for this video without FFmpeg. Install FFmpeg and try again.",
            )

        title = info.get("title") or "video"

        # Optional compatibility pass: only when not in fast mode
        if ffmpeg_available and not fast:
            compatible_path = os.path.join(temp_dir, f"{title}.compat.mp4")
            try:
                # -movflags +faststart moves moov atom to the beginning
                # Baseline@3.1, CFR 30 fps, yuv420p, AAC-LC stereo for maximum browser/WMP compatibility
                print("[ffmpeg] Transcoding for compatibility...")
                subprocess.run(
                    [
                        FFMPEG_EXE or "ffmpeg",
                        "-y",
                        "-i",
                        file_path,
                        # map first video and first audio only; drop subs/data/metadata/chapters
                        "-map", "0:v:0",
                        "-map", "0:a:0?",
                        "-sn",
                        "-dn",
                        "-map_metadata", "-1",
                        "-map_chapters", "-1",
                        # video
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-profile:v", "baseline",
                        "-level:v", "3.1",
                        "-r", "30",
                        "-g", "60",
                        "-keyint_min", "60",
                        "-sc_threshold", "0",
                        # audio
                        "-c:a", "aac",
                        "-b:a", "128k",
                        "-ar", "44100",
                        "-ac", "2",
                        # container flags
                        "-movflags", "+faststart",
                        compatible_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                print("[ffmpeg] Transcode done.")
                # Replace file_path with the compatible output and remove the original
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                file_path = compatible_path
            except subprocess.CalledProcessError:
                # If transcode fails, fall back to the downloaded file
                pass

        return temp_dir, file_path, title
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.exception("Unhandled server error during download for url=%s", url)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


def _download_audio_to_tmp(
    url: str,
    progress_id: str | None = None,
    cookies_file: str | None = None,
    codec: str = "mp3",
) -> Tuple[str, str, str]:
    """Download audio and extract to the requested codec (mp3 or wav). Requires FFmpeg."""
    if not _has_ffmpeg():
        raise HTTPException(status_code=415, detail="FFmpeg required for mp3/wav extraction.")

    temp_dir = tempfile.mkdtemp(prefix="ytvdl_a_")

    ydl_opts = {
        "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
        "restrictfilenames": True,
        "format": "bestaudio/best",
        "ffmpeg_location": FFMPEG_EXE,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": codec,
                "preferredquality": "0",
            }
        ],
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    if progress_id:
        ydl_opts["progress_hooks"] = [_make_progress_hook(progress_id)]
    else:
        ydl_opts["progress_hooks"] = [_progress_printer]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            prepared = ydl.prepare_filename(info)
            base, _ = os.path.splitext(prepared)
            desired = base + f".{codec}"
            # yt-dlp may normalize extensions (e.g., .m4a -> .mp3)
            candidates = [desired, base + ".mp3", base + ".wav", prepared]
            file_path = next((p for p in candidates if os.path.exists(p)), None)
            if not file_path:
                shutil.rmtree(temp_dir, True)
                raise HTTPException(status_code=500, detail="Audio extraction finished but file not found.")
            title = info.get("title") or "audio"
            return temp_dir, file_path, title
    except yt_dlp.utils.DownloadError as e:
        shutil.rmtree(temp_dir, True)
        logger.exception("yt-dlp audio download error for url=%s", url)
        raise HTTPException(status_code=400, detail=f"Download error: {str(e)}")
    except Exception as e:
        shutil.rmtree(temp_dir, True)
        logger.exception("Unhandled error during audio download for url=%s", url)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


def _run_download_job(
    progress_id: str,
    base_url: str,
    url: str,
    ext: str,
    height: int | None,
    link_only: bool,
    cookies_path: str | None,
    cookies_owned: bool,
    fast: bool,
) -> None:
    """Background worker that performs the download/transcode and publishes a link."""
    try:
        # Branch for audio-only
        if ext in {"mp3", "wav"}:
            temp_dir, file_path, title = _download_audio_to_tmp(url, progress_id, cookies_path, codec=ext)
        else:
            temp_dir, file_path, title = _download_video_to_tmp(url, progress_id, cookies_path, target_height=height, fast=fast)

        # Optional transcode if strictly required
        out_path = file_path
        if ext not in {"mp3", "wav"} and _has_ffmpeg() and ((ext != "mp4") or (height is not None and not fast)):
            if progress_id in JOB_PROGRESS:
                JOB_PROGRESS[progress_id].update({"stage": "transcoding"})
            suffix = f".{height}p" if height else ""
            out_path = os.path.join(os.path.dirname(file_path), f"{title}{suffix}.{ext}")
            try:
                cmd = _build_ffmpeg_cmd(file_path, out_path, ext, height)
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                file_path = out_path
            except subprocess.CalledProcessError:
                logger.exception("ffmpeg transcode failed: src=%s dst=%s ext=%s height=%s", file_path, out_path, ext, height)
                file_path = out_path if os.path.exists(out_path) else file_path

        filename = os.path.basename(file_path)
        # For background mode we always publish a link
        name, ext_actual = os.path.splitext(filename)
        dest_name = filename
        dest_path = os.path.join(DOWNLOAD_DIR, dest_name)
        if os.path.exists(dest_path):
            dest_name = f"{name}-{uuid.uuid4().hex[:8]}{ext_actual}"
            dest_path = os.path.join(DOWNLOAD_DIR, dest_name)
        shutil.move(file_path, dest_path)
        encoded_name = urllib.parse.quote(dest_name)
        file_url = base_url.rstrip("/") + f"/files/{encoded_name}"
        if progress_id in JOB_PROGRESS:
            JOB_PROGRESS[progress_id].update({
                "stage": "finished",
                "status": "completed",
                "pct": 100.0,
                "file_url": file_url,
                "filename": dest_name,
            })
    except Exception:
        logger.exception("Background job failed for url=%s", url)
        JOB_PROGRESS[progress_id] = {
            "stage": "error",
            "status": "failed",
            "pct": 0.0,
            "error": "Background job failed. See server logs.",
        }
    finally:
        try:
            # Best-effort cleanup
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, True)
        except Exception:
            pass
        if cookies_owned and cookies_path:
            _delete_path_safely(cookies_path)


@app.get("/")
def root():
    return JSONResponse({"status": "ok", "message": "Use /download?url=... to fetch MP4"})

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/healthz")
def healthz():
    return JSONResponse({"status": "ok"})

@app.head("/healthz")
def healthz_head():
    return Response(status_code=200)
@app.get("/progress/{progress_id}")
def get_progress(progress_id: str):
    data = JOB_PROGRESS.get(progress_id)
    if not data:
        raise HTTPException(status_code=404, detail="Progress id not found")
    return JSONResponse(data)



def _build_ffmpeg_cmd(src_path: str, dst_path: str, ext: str, height: int | None) -> list[str]:
    """Build an ffmpeg command to transcode to the requested container/height.

    - mp4: H.264 Baseline@3.1, AAC-LC stereo, 30fps, yuv420p, +faststart
    - webm: VP9 + Opus stereo, 30fps
    - mkv: H.264 + AAC stereo, 30fps
    """
    scale_args: list[str] = []
    if height:
        # Keep width divisible by 2 while preserving aspect ratio
        scale_args = ["-vf", f"scale=-2:{height}"]

    if ext == "mp4":
        return [
            FFMPEG_EXE or "ffmpeg", "-y", "-i", src_path,
            "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
            "-map_metadata", "-1", "-map_chapters", "-1",
            *scale_args,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level:v", "3.1",
            "-r", "30", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", dst_path,
        ]
    if ext == "webm":
        return [
            FFMPEG_EXE or "ffmpeg", "-y", "-i", src_path,
            "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
            "-map_metadata", "-1", "-map_chapters", "-1",
            *scale_args,
            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "33", "-r", "30",
            "-pix_fmt", "yuv420p",
            "-c:a", "libopus", "-b:a", "128k", "-ac", "2",
            dst_path,
        ]
    # mkv fallback
    return [
        FFMPEG_EXE or "ffmpeg", "-y", "-i", src_path,
        "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
        "-map_metadata", "-1", "-map_chapters", "-1",
        *scale_args,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-profile:v", "main", "-level:v", "4.0",
        "-r", "30", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        dst_path,
    ]


@app.get("/download")
def download(
    request: Request,
    url: str = Query(..., description="YouTube video URL"),
    ext: str = Query("mp4", description="Output container: mp4 | webm | mkv | mp3 | wav"),
    height: int | None = Query(None, description="Target height in px: 240/360/480/720/1080"),
    disposition: str = Query("attachment", description="inline or attachment"),
    link_only: bool = Query(True, description="Return JSON with a hosted file link instead of file data"),
    progress_id: str | None = Query(None, description="Client-provided id to poll progress at /progress/{id}"),
    cookies_b64: str | None = Query(None, description="Optional base64-encoded Netscape cookies.txt for yt-dlp"),
    fast: bool = Query(True, description="Skip re-encoding when possible; fastest path"),
    async_job: bool = Query(True, description="Run download in background and return 202 immediately"),
    background_tasks: BackgroundTasks = None,
):
    # Ensure a progress id for clients that want to poll
    if not progress_id:
        progress_id = uuid.uuid4().hex
    JOB_PROGRESS[progress_id] = {"stage": "starting", "status": "running", "pct": 0.0}

    try:
        logger.info(
            "Request /download: url=%s ext=%s height=%s disposition=%s link_only=%s progress_id=%s",
            url, ext, height, disposition, link_only, progress_id,
        )
    except Exception:
        # Avoid breaking the request due to logging issues
        pass

    # Prepare optional cookies file
    cookies_path: str | None = None
    cookies_owned: bool = False
    if cookies_b64:
        try:
            decoded = base64.b64decode(cookies_b64)
            import tempfile as _tf
            cookies_tmp = _tf.NamedTemporaryFile(prefix="ytvdl_req_cookies_", suffix=".txt", delete=False)
            with cookies_tmp as f:
                f.write(decoded)
            cookies_path = cookies_tmp.name
            cookies_owned = True
            logger.info("/download: using cookies from request param")
        except Exception:
            logger.exception("/download: failed decoding/writing cookies_b64 param")
            raise HTTPException(status_code=400, detail="Invalid cookies_b64 parameter")
    elif COOKIES_FILE:
        cookies_path = COOKIES_FILE
        logger.info("/download: using cookies from env file")

    # If async job, start a background thread and return 202 immediately
    if async_job:
        base_url = str(request.base_url)
        thread = threading.Thread(
            target=_run_download_job,
            kwargs={
                "progress_id": progress_id,
                "base_url": base_url,
                "url": url,
                "ext": ext,
                "height": height,
                "link_only": True,
                "cookies_path": cookies_path,
                "cookies_owned": cookies_owned,
                "fast": fast,
            },
            daemon=True,
        )
        thread.start()
        return JSONResponse({
            "status": "accepted",
            "progress_id": progress_id,
            "progress_url": str(request.base_url).rstrip("/") + f"/progress/{progress_id}",
            "note": "Job started. Poll progress_url until status=completed to get file_url.",
        }, status_code=202)

    # Branch: audio vs video for sync mode
    if ext in {"mp3", "wav"}:
        temp_dir, file_path, title = _download_audio_to_tmp(url, progress_id, cookies_path, codec=ext)
    else:
        temp_dir, file_path, title = _download_video_to_tmp(url, progress_id, cookies_path, target_height=height, fast=fast)

    # Schedule cleanup after response is sent
    if background_tasks is not None:
        background_tasks.add_task(shutil.rmtree, temp_dir, True)

    # Validate ext
    ext = (ext or "mp4").lower()
    if ext not in {"mp4", "webm", "mkv", "mp3", "wav"}:
        raise HTTPException(status_code=400, detail="Invalid ext. Use mp4, webm, mkv, mp3, or wav.")

    # If no FFmpeg, only mp4 without scaling is supported; audio extraction requires FFmpeg
    if not _has_ffmpeg():
        if ext in {"mp3", "wav"}:
            raise HTTPException(status_code=415, detail="FFmpeg required for mp3/wav extraction.")
        if ext != "mp4" or height is not None:
            raise HTTPException(status_code=415, detail="FFmpeg required for custom ext/height.")

    # Transcode to requested ext/height if needed
    out_path = file_path
    # Audio path: handled by yt-dlp extractor already; skip video transcode
    if ext in {"mp3", "wav"}:
        pass
    # Only transcode when necessary: custom container (not mp4) or explicit height AND fast==False
    elif _has_ffmpeg() and ((ext != "mp4") or (height is not None and not fast)):
        # Mark transcoding stage for visibility
        if progress_id in JOB_PROGRESS:
            JOB_PROGRESS[progress_id].update({"stage": "transcoding"})
        # Build output path with resolution suffix if provided
        suffix = f".{height}p" if height else ""
        out_path = os.path.join(os.path.dirname(file_path), f"{title}{suffix}.{ext}")
        try:
            cmd = _build_ffmpeg_cmd(file_path, out_path, ext, height)
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            try:
                os.remove(file_path)
            except Exception:
                pass
            file_path = out_path
        except subprocess.CalledProcessError:
            logger.exception("ffmpeg transcode failed: src=%s dst=%s ext=%s height=%s", file_path, out_path, ext, height)
            # If transcode fails, fall back to original
            file_path = out_path if os.path.exists(out_path) else file_path

    # Prepare response
    filename = os.path.basename(file_path)
    if ext == "mp3":
        media_type = "audio/mpeg"
    elif ext == "wav":
        media_type = "audio/wav"
    else:
        media_type = "video/mp4" if ext == "mp4" else ("video/webm" if ext == "webm" else "video/x-matroska")

    # If link_only, move the file to the public downloads dir and return JSON link
    if link_only:
        name, ext_actual = os.path.splitext(filename)
        dest_name = filename
        dest_path = os.path.join(DOWNLOAD_DIR, dest_name)
        if os.path.exists(dest_path):
            dest_name = f"{name}-{uuid.uuid4().hex[:8]}{ext_actual}"
            dest_path = os.path.join(DOWNLOAD_DIR, dest_name)
        shutil.move(file_path, dest_path)
        # Build a URL with percent-encoded filename so spaces/special chars work in browsers
        encoded_name = urllib.parse.quote(dest_name)
        file_url = str(request.base_url).rstrip("/") + f"/files/{encoded_name}"
        # Mark finished and attach file url in progress map
        if progress_id in JOB_PROGRESS:
            JOB_PROGRESS[progress_id].update({
                "stage": "finished",
                "status": "completed",
                "pct": 100.0,
                "file_url": file_url,
                "filename": dest_name,
            })
        return JSONResponse({
            "url": file_url,
            "filename": dest_name,
            "media_type": media_type,
            "filesize_bytes": os.path.getsize(dest_path) if os.path.exists(dest_path) else None,
            "format": (
                {
                    "container": ext,
                    "audio": "MP3 (ffmpeg), VBR/auto",
                } if ext == "mp3" else (
                {
                    "container": ext,
                    "audio": "WAV (PCM), 44.1kHz",
                } if ext == "wav" else {
                    "container": ext,
                    "video": "H.264 (libx264) baseline@3.1, yuv420p, 30fps",
                    "audio": "AAC-LC, 128k, 44.1kHz, stereo",
                })
            ),
            "progress_id": progress_id,
        })

    disp = "inline" if str(disposition).lower() == "inline" else "attachment"
    headers = {"Content-Disposition": f'{disp}; filename="{filename}"'}
    return FileResponse(path=file_path, media_type=media_type, headers=headers)


# Removed deprecated on_event handlers in favor of lifespan above


if __name__ == "__main__":
    import uvicorn

    # Disable reload by default to avoid Windows/Python 3.13 reload issues.
    # Opt-in by setting environment variable RELOAD=1
    reload_enabled = os.getenv("RELOAD", "0") in ("1", "true", "TRUE", "True")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=reload_enabled,
    )
