
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


if __name__ == '__main__':
    from src.train.train_finetune import train
    train()
