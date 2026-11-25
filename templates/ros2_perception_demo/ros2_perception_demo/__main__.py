"""Select a platform executable; all processing is owned by that module."""
import argparse
from robotic_inference import run_pipeline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args,extra=parser.parse_known_args()
    return run_pipeline(args.config,extra).returncode

if __name__=='__main__': raise SystemExit(main())
