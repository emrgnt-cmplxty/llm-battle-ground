import argparse


def common_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Logging args
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level, e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    # Data args
    parser.add_argument(
        "--in-dir",
        type=str,
        default=None,
        help="Path to pull the dataset to for the script run.",
    )
    parser.add_argument(
        "--in-file-name",
        type=str,
        default=None,
        help="Name of the file to pull the dataset to for the script run.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Path to save the dataset to for this script",
    )
    parser.add_argument(
        "--out-file-name",
        type=str,
        default=None,
        help="Name of the file to pull the dataset to for the script run.",
    )

    # Config args
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Default provider for LLM use, e.g. `openai`.",
    )

    parser.add_argument(
        "--run-mode",
        type=str,
        default=None,
        help="Mode to run the experiment in, e.g. `zero-shot`, `finetune`....",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for the script the script run, e.g. `gpt-4-0613`.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature to use for the script.",
    )
    parser.add_argument(
        "--n-pass",
        type=int,
        default=None,
        help="Number of calculations to make for the script figure of merit.",
    )

    return parser
