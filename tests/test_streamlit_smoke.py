import subprocess
import time
import shutil
import pytest

try:
    import requests
except Exception:
    requests = None


def test_streamlit_serves_homepage():
    # Require streamlit CLI and requests to run the smoke test locally (CI may opt-in)
    if shutil.which("streamlit") is None:
        pytest.skip("streamlit CLI not installed")
    if requests is None:
        pytest.skip("requests not installed")

    port = 8502
    url = f"http://localhost:{port}"
    cmd = ["streamlit", "run", "app.py", "--server.port", str(port), "--server.headless", "true"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        timeout = 20
        for _ in range(timeout):
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200 and "AI Music Visualizer" in r.text:
                    return
            except Exception:
                time.sleep(1)
        pytest.fail("Streamlit homepage didn't become available or content did not match")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
