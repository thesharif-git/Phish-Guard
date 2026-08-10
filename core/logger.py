import logging
import sys

# Configure logging
def setup_logging():
    # Ensure UTF-8 stream handler on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    file_handler = logging.FileHandler("server.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[file_handler, stream_handler]
    )
    
logger = logging.getLogger("phishguard")
