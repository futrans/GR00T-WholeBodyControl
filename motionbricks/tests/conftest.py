import sys
from pathlib import Path


MOTIONBRICKS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTIONBRICKS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONBRICKS_ROOT))
