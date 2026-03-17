import os
import subprocess
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, HTTPException, Request, status
from starlette.datastructures import UploadFile as StarletteUploadFile
import json
import requests 
import asyncio
import re
import secrets
import unicodedata
import wave

app = FastAPI()
# Enforce a strict 5MB limit on the backend too
MAX_FILE_SIZE = 5 * 1024 * 1024 
MAX_MULTIPART_BODY_SIZE = 6 * 1024 * 1024
MAX_UPLOAD_SECONDS = 30
MAX_BODY_CHUNK_IDLE_SECONDS = 5
request_slot = asyncio.Queue(maxsize=1)


def sanitize_whisper_text(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKC", text)
    # Drop ASCII control chars, DEL, common zero-width, and bidi override/isolate chars.
    # Keep standard punctuation to preserve transcript readability/meaning.
    blocked_chars = (
        [chr(i) for i in range(0x00, 0x20)]
        + [chr(0x7F)]
        + ["\u200B", "\u200C", "\u200D", "\uFEFF"]
        + [chr(i) for i in range(0x202A, 0x202F)]
        + [chr(i) for i in range(0x2066, 0x206A)]
    )
    translation_table = {ord(ch): None for ch in blocked_chars}
    cleaned = normalized_text.translate(translation_table)
    return re.sub(r"\s+", " ", cleaned).strip()


def status_id_to_filename(status_id: str) -> str:
    safe_status_id = re.sub(r"[^A-Za-z0-9._-]", "_", status_id).strip("._-")
    if not safe_status_id:
        safe_status_id = "unknown_status"
    return f"{safe_status_id}.json"


def status_id_to_wav_filename(status_id: str) -> str:
    safe_status_id = re.sub(r"[^A-Za-z0-9._-]", "_", status_id).strip("._-")
    if not safe_status_id:
        safe_status_id = "unknown_status"
    return f"{safe_status_id}.wav"


def get_user_voiceclone_dir() -> str:
    output_dir = os.path.join(os.path.dirname(__file__), "voiceclones", "users")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def is_english_alphanumeric(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9]+", value))


def is_valid_api_key(auth_header: str) -> bool:
    expected_api_key = os.getenv("VOICE_CLONE_API_KEY")
    if not expected_api_key or not is_english_alphanumeric(expected_api_key):
        return False

    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    if not is_english_alphanumeric(token):
        return False

    return secrets.compare_digest(token, expected_api_key)


def write_json_file(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as response_file:
        json.dump(payload, response_file, ensure_ascii=False, indent=2)


def raise_http_error(status_code: int, detail: str, context: str) -> None:
    print(f"[HTTP Error] {context} (status={status_code}): {detail}", flush=True)
    raise HTTPException(status_code=status_code, detail=detail)


def post_groq_transcription(file_path: str) -> requests.Response:
    with open(file_path, "rb") as audio_file:
        return requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}",
            },
            files={
                "file": (os.path.basename(file_path), audio_file, "audio/wav"),
            },
            data={
                "model": "whisper-large-v3",
            },
            timeout=5,
        )


def rewrite_wav_with_essential_chunks(input_path: str, output_path: str) -> None:
    """
    Rebuild WAV so output contains only canonical PCM structure.
    This drops ancillary RIFF chunks and keeps only the audio stream.
    """
    with wave.open(input_path, "rb") as src:
        with wave.open(output_path, "wb") as dst:
            dst.setnchannels(src.getnchannels())
            dst.setsampwidth(src.getsampwidth())
            dst.setframerate(src.getframerate())
            dst.setcomptype(src.getcomptype(), src.getcompname())

            chunk_frames = 4096
            while True:
                frames = src.readframes(chunk_frames)
                if not frames:
                    break
                dst.writeframes(frames)


def verify_safe_wav(file_path: str) -> bool:
    """Verify sanitized WAV matches strict format and duration constraints."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print("[Verify Error] File is missing or empty.", flush=True)
        return False

    try:
        with wave.open(file_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()  # 2 bytes = 16-bit
            frames = wf.getnframes()

            # Calculate duration in seconds.
            duration = frames / float(sample_rate) if sample_rate > 0 else 0.0

            if channels != 1:
                print(f"[Verify Error] Expected mono (1 channel), got {channels}", flush=True)
                return False
            if sample_rate != 32000:
                print(f"[Verify Error] Expected 32kHz, got {sample_rate}", flush=True)
                return False
            if sample_width != 2:
                print(f"[Verify Error] Expected 16-bit audio, got {sample_width * 8}-bit", flush=True)
                return False
            if duration > 10.1:
                print(f"[Verify Error] Audio exceeds 10 seconds (Actual: {duration:.2f}s)", flush=True)
                return False

            return True
    except wave.Error as e:
        print(f"[Verify Error] Not a valid WAV file: {e}", flush=True)
        return False


class RequestBodyTooLarge(Exception):
    """Raised when incoming request body exceeds allowed byte limit."""


class RequestBodyReadTimeout(Exception):
    """Raised when incoming request body is too slow or exceeds read deadline."""


class BodySizeLimitMiddleware:
    def __init__(
        self,
        app,
        max_body_size: int,
        method: str,
        path: str,
        max_body_read_seconds: int | float,
        chunk_idle_timeout_seconds: int | float,
    ):
        self.app = app
        self.max_body_size = max_body_size
        self.method = method.upper()
        self.path = path
        self.max_body_read_seconds = float(max_body_read_seconds)
        self.chunk_idle_timeout_seconds = float(chunk_idle_timeout_seconds)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("method", "").upper() != self.method or scope.get("path", "") != self.path:
            await self.app(scope, receive, send)
            return

        # Fast-fail based on declared content length when available.
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_size:
                    await self._send_413(send)
                    return
            except ValueError:
                await self._send_400(send)
                return

        bytes_seen = 0
        response_started = False
        loop = asyncio.get_running_loop()
        body_read_deadline = loop.time() + self.max_body_read_seconds

        async def tracked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def limited_receive():
            nonlocal bytes_seen
            remaining = body_read_deadline - loop.time()
            if remaining <= 0:
                raise RequestBodyReadTimeout()

            # Apply both a total body deadline and per-chunk idle timeout.
            wait_budget = min(remaining, self.chunk_idle_timeout_seconds)
            try:
                message = await asyncio.wait_for(receive(), timeout=wait_budget)
            except asyncio.TimeoutError:
                raise RequestBodyReadTimeout()

            if message["type"] == "http.request":
                bytes_seen += len(message.get("body", b""))
                if bytes_seen > self.max_body_size:
                    raise RequestBodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            # Avoid ASGI protocol violations by never sending a second response start.
            if not response_started:
                print("[Middleware Error] Request body exceeded configured size limit.", flush=True)
                await self._send_413(send)
        except RequestBodyReadTimeout:
            if not response_started:
                print("[Middleware Error] Request body read timed out.", flush=True)
                await self._send_408(send)

    async def _send_413(self, send):
        print("[Middleware Error] Responding with 413: request body too large.", flush=True)
        body = json.dumps({"detail": "Request body too large. Limit is 6MB."}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _send_400(self, send):
        print("[Middleware Error] Responding with 400: invalid Content-Length header.", flush=True)
        body = json.dumps({"detail": "Invalid Content-Length header."}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _send_408(self, send):
        print("[Middleware Error] Responding with 408: upload timed out.", flush=True)
        body = json.dumps({"detail": "Upload timed out."}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 408,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


app.add_middleware(
    BodySizeLimitMiddleware,
    max_body_size=MAX_MULTIPART_BODY_SIZE,
    method="POST",
    path="/v1/voice/clone",
    max_body_read_seconds=MAX_UPLOAD_SECONDS,
    chunk_idle_timeout_seconds=MAX_BODY_CHUNK_IDLE_SECONDS,
)

@app.post("/v1/voice/clone")
async def process_voice_clone(request: Request):
    slot_acquired = False
    file = None
    auth_header = request.headers.get("authorization", "")
    status_id = request.headers.get("x-status-id")
    content_type = request.headers.get("content-type", "")

    if not is_valid_api_key(auth_header):
        raise_http_error(401, "Missing or invalid Authorization header.", "Authorization validation")
    if not status_id:
        raise_http_error(400, "Missing X-Status-Id header.", "Header validation")
    if "multipart/form-data" not in content_type:
        raise_http_error(415, "Expected multipart/form-data request.", "Content-Type validation")

    # Atomically reserve the only request slot before accepting upload work.
    # This avoids wasting ingress bandwidth/disk IO on requests that cannot run.
    try:
        request_slot.put_nowait(object())
        slot_acquired = True
    except asyncio.QueueFull:
        raise_http_error(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "A process is already running. Please try again later.",
            "Request slot acquisition",
        )

    try:
        try:
            form = await request.form(
                max_files=1,
                max_fields=10,
                max_part_size=MAX_FILE_SIZE,
            )
        except HTTPException as e:
            print(f"[Request Error] Multipart parsing HTTPException: {e.detail}", flush=True)
            raise
        except Exception as e:
            print(f"[Request Error] Malformed multipart form-data: {e}", flush=True)
            raise_http_error(400, "Malformed multipart form-data.", "Multipart parsing")
        form_keys = list(form.keys())
        form_types = {k: type(v).__name__ for k, v in form.items()}
        print(f"[400-debug] request: multipart keys={form_keys} part_types={form_types}", flush=True)
        file = form.get("file")
        if not isinstance(file, StarletteUploadFile):
            if "file" in form:
                print(
                    f"[400-debug] request: `file` part exists but has type={type(form.get('file')).__name__}",
                    flush=True,
                )
            else:
                print("[400-debug] request: multipart form missing `file` field", flush=True)
            raise_http_error(400, "Missing `file` in multipart form-data.", "Multipart field validation")

        # 1. Reject non-audio MIME types immediately (Basic check)
        if not file.content_type.startswith("audio/"):
            raise_http_error(400, "Invalid file type.", "MIME type validation")

        untrusted_path = None
        safe_path = None
        minimal_safe_path = None
        persisted_safe_wav_path = None
        try:
            # 2. Use tempfile to create secure, randomized file names
            # This prevents directory traversal attacks (e.g., if a user names a file "../../etc/passwd")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as untrusted_file:
                untrusted_path = untrusted_file.name

                # Read the file in 1MB chunks so we don't blow up the RAM
                size = 0
                header = bytearray()
                while True:
                    content = await file.read(1024 * 1024)
                    if not content:
                        break
                    size += len(content)
                    if size > MAX_FILE_SIZE:
                        raise_http_error(413, "File exceeds 5MB limit.", "Upload size validation")
                    if len(header) < 12:
                        needed = 12 - len(header)
                        header.extend(content[:needed])
                        # Fail fast: validate as soon as we have enough bytes from the first chunk(s).
                        if len(header) >= 4 and header[0:4] != b"RIFF":
                            raise_http_error(400, "Invalid WAV header. Expected RIFF/WAVE format.", "WAV header validation")
                        if len(header) >= 12 and header[8:12] != b"WAVE":
                            raise_http_error(400, "Invalid WAV header. Expected RIFF/WAVE format.", "WAV header validation")
                    untrusted_file.write(content)

            # Validate WAV container signature: RIFF....WAVE
            if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise_http_error(400, "Invalid WAV header. Expected RIFF/WAVE format.", "WAV container validation")

            # Define where the clean file will go
            safe_path = untrusted_path.replace(".wav", "_clean.wav")
            minimal_safe_path = untrusted_path.replace(".wav", "_minimal.wav")
            user_voiceclone_dir = get_user_voiceclone_dir()
            persisted_safe_wav_path = os.path.join(
                user_voiceclone_dir,
                status_id_to_wav_filename(status_id),
            )

            # 3. The Decontamination Chamber (FFmpeg)
            # -y: overwrite output silently
            # -i: input file
            # -ac 1: force mono audio (1 channel)
            # -ar 22050: force 22.05kHz sample rate (Change to match your TTS model!)
            # -t 15: hard crop to 15 seconds max (Prevents processing hour-long files)
            command = [
                "ffmpeg", "-nostdin", "-y",
                "-protocol_whitelist", "file",
                "-f", "wav",
                "-i", untrusted_path,
                "-map_metadata", "-1",
                "-map_chapters", "-1",
                # Step 1: Up-sample to 48kHz.
                "-af", "aresample=48000,volume=0.99",
                "-ar", "32000",
                "-ac", "1",
                "-t", "10",
                "-c:a", "pcm_s16le",
                "-f", "wav",
                # Force a 1% volume reduction. Unnoticeable to humans and the TTS model, 
                # but recalculates every single 16-bit integer, obliterating LSB data.
                safe_path
            ]
            # ffmpeg -i untrusted.wav -c:a libmp3lame -qscale:a 2 -f mp3 - | ffmpeg -i - -c:a pcm_s16le -ar 22050 -ac 1 clean_audio.wav
            # Run the command with a timeout. If a malicious file tries to hang FFmpeg, 
            # it kills the process after 10 seconds.
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10
                )
            except subprocess.CalledProcessError as e:
                print(f"[FFmpeg Error] Command failed: {e}", flush=True)
                # If FFmpeg fails, the file was likely corrupted or not actually an audio file.
                raise_http_error(400, "Audio file is corrupted or unreadable.", "FFmpeg processing")
            except subprocess.TimeoutExpired as e:
                print(f"[FFmpeg Error] Processing timed out: {e}", flush=True)
                raise_http_error(408, "Audio processing timed out.", "FFmpeg timeout")
            try:
                await asyncio.to_thread(rewrite_wav_with_essential_chunks, safe_path, minimal_safe_path)
                await asyncio.to_thread(os.replace, minimal_safe_path, safe_path)
            except (wave.Error, OSError) as e:
                print(f"[WAV Normalize Error] Failed to normalize WAV container: {e}", flush=True)
                raise_http_error(400, "Failed to normalize WAV container.", "WAV normalization")
            try:
                await asyncio.to_thread(shutil.copyfile, safe_path, persisted_safe_wav_path)
            except OSError as e:
                print(f"[WAV Persist Error] Failed to write sanitized WAV file: {e}", flush=True)
                raise_http_error(500, "Failed to write sanitized WAV file.", "WAV persistence")

            # 4. INFERENCE: The file is now safe. Pass `safe_path` to your TTS model.
            # result_audio = my_tts_model.infer(voice_sample=safe_path, text="Hello world")
            
            try:
                if not verify_safe_wav(safe_path):
                    raise_http_error(400, "Sanitized WAV failed strict verification checks.", "WAV verification")
                response = await asyncio.to_thread(post_groq_transcription, safe_path)
                response.raise_for_status()
                try:
                    groq_payload = response.json()
                except ValueError as e:
                    print(f"[Groq Error] Invalid transcription JSON response: {e}", flush=True)
                    raise_http_error(502, "Invalid transcription response from Groq.", "Groq JSON parsing")

                raw_text = groq_payload.get("text", "")
                sanitized_text = sanitize_whisper_text(raw_text if isinstance(raw_text, str) else str(raw_text))
                whisper_response = {
                    "text": sanitized_text,
                    "x_groq": {
                        "id": (groq_payload.get("x_groq") or {}).get("id", "")
                    },
                }
                try:
                    response_path = os.path.join(
                        user_voiceclone_dir,
                        status_id_to_filename(status_id),
                    )
                    await asyncio.to_thread(write_json_file, response_path, whisper_response)
                except OSError as e:
                    print(f"[Groq Persist Error] Failed to write Whisper response file: {e}", flush=True)
                    raise_http_error(500, "Failed to write Groq Whisper response file.", "Whisper response persistence")
            except requests.RequestException as e:
                print(f"[Groq Request Error] Backend error: {e}", flush=True)
                raise_http_error(502, "Backend error.", "Groq request")
            

            return {"status": "success", "message": "Voice processed and cloned successfully!"}

        except subprocess.CalledProcessError as e:
            print(f"[FFmpeg Error] Command failed (outer handler): {e}", flush=True)
            # If FFmpeg fails, the file was likely corrupted or not actually an audio file
            raise_http_error(400, "Audio file is corrupted or unreadable.", "FFmpeg outer handler")
        except subprocess.TimeoutExpired as e:
            print(f"[FFmpeg Error] Processing timed out (outer handler): {e}", flush=True)
            raise_http_error(408, "Audio processing timed out.", "FFmpeg timeout outer handler")
        
        finally:
            # 5. CRITICAL CLEANUP: Always delete the temporary files!
            # If you forget this, your RunPod server disk will fill up and crash.
            if untrusted_path and os.path.exists(untrusted_path):
                os.remove(untrusted_path)
            if safe_path and os.path.exists(safe_path):
                os.remove(safe_path)
            if minimal_safe_path and os.path.exists(minimal_safe_path):
                os.remove(minimal_safe_path)
    finally:
        if isinstance(file, UploadFile):
            try:
                await file.close()
            except Exception as e:
                print(f"[Cleanup Error] Failed to close uploaded file: {e}", flush=True)
                # Cleanup failure should never mask the primary request error.
                pass
        if slot_acquired:
            try:
                request_slot.get_nowait()
            except asyncio.QueueEmpty:
                pass
        
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("clone_point:app", host="0.0.0.0", port=8000, reload=False)