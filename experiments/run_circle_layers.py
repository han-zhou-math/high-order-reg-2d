import argparse
from .config import RESULT_ROOT
from .local_drivers import run_circle_layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute the production study")
    args = parser.parse_args()
    if not args.run: parser.error("pass --run to execute")
    run_circle_layers(RESULT_ROOT / "circle_layers")

