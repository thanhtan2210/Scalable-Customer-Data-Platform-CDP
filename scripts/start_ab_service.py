import socket
import sys
from pathlib import Path
import uvicorn

# Ensure project root is on sys.path so imports like `src.api` work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_free_port(host='127.0.0.1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()
    return port


if __name__ == '__main__':
    host = '127.0.0.1'
    port = find_free_port(host)
    Path('reports').mkdir(parents=True, exist_ok=True)
    Path('reports/ab_service_port.txt').write_text(str(port))
    print(f'Starting ab_service on {host}:{port}')
    uvicorn.run('src.api.ab_service:app', host=host,
                port=port, log_level='info')
