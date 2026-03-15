"""
Run the full recommendation pipeline end-to-end.

Usage:
    python run_pipeline.py              # Run all stages
    python run_pipeline.py preprocess   # Run only preprocessing
    python run_pipeline.py retrieval    # Run only retrieval training
    python run_pipeline.py faiss        # Run only FAISS index building
    python run_pipeline.py ranking      # Run only ranking training
    python run_pipeline.py evaluate     # Run only evaluation
    python run_pipeline.py serve        # Start API server
"""

import sys
import subprocess


STAGES = {
    "preprocess": "data/preprocess.py",
    "retrieval": "retrieval/train.py",
    "faiss": "retrieval/faiss_index.py",
    "ranking": "ranking/train.py",
    "evaluate": "evaluation/evaluate.py",
}


def run_stage(name, script):
    print(f"\n{'='*60}")
    print(f"  Stage: {name}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script], cwd=".")
    if result.returncode != 0:
        print(f"\nERROR: Stage '{name}' failed with code {result.returncode}")
        sys.exit(1)


def main():
    args = sys.argv[1:]

    if not args:
        # Run all stages
        for name, script in STAGES.items():
            run_stage(name, script)
        print("\n\nPipeline complete! Start the API with: python run_pipeline.py serve")
    elif args[0] == "serve":
        print("Starting API server...")
        subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app",
                        "--reload", "--host", "0.0.0.0", "--port", "8000"])
    elif args[0] in STAGES:
        run_stage(args[0], STAGES[args[0]])
    else:
        print(f"Unknown stage: {args[0]}")
        print(f"Available stages: {', '.join(list(STAGES.keys()) + ['serve'])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
