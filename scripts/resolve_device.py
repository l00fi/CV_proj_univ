"""Resolve training device for Docker entrypoint (stdout = device id, stderr = description)."""

from __future__ import annotations

import argparse
import sys

from poker_yolo.device import describe_device, detect_training_device, strict_cuda_required


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit with code 1 if CUDA is not available (also set via REQUIRE_CUDA=1)",
    )
    args = parser.parse_args(argv)

    strict = args.require_gpu or strict_cuda_required()
    try:
        device = detect_training_device(strict_cuda=strict)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(describe_device(device), file=sys.stderr)
    if args.require_gpu and device == "cpu":
        print(
            "Hint: enable GPU in Docker Desktop or install NVIDIA Container Toolkit.",
            file=sys.stderr,
        )
        return 1

    print(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
