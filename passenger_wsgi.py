# Entry point for cPanel "Setup Python App" (Phusion Passenger, WSGI) on GoDaddy shared hosting.
# FastAPI is ASGI, so it is wrapped with a2wsgi. Passenger does not run ASGI lifespan events,
# so the startup work from main._lifespan (cookies from env, periodic cleanup) is done here.
import base64
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

import main  # noqa: E402
from a2wsgi import ASGIMiddleware  # noqa: E402

_cookies_b64 = os.getenv("YTDLP_COOKIES_B64")
if _cookies_b64 and not main.COOKIES_FILE:
    try:
        decoded = main._decode_cookies_payload(_cookies_b64)
        fd, path = tempfile.mkstemp(prefix="ytvdl_cookies_", suffix=".txt")
        with os.fdopen(fd, "wb") as f:
            f.write(decoded)
        main.COOKIES_FILE = path
    except Exception:
        main.logger.exception("passenger: failed to load YTDLP_COOKIES_B64")


def _cleanup_forever():
    while True:
        try:
            main._cleanup_downloads_once(time.time())
        except Exception:
            main.logger.exception("passenger: cleanup failed")
        time.sleep(main.CLEANUP_INTERVAL_SECONDS)


threading.Thread(target=_cleanup_forever, name="ytvdl-cleanup", daemon=True).start()

application = ASGIMiddleware(main.app)
