import argparse
import sys

from config         import MODEL_ID, BLOOMS_VERBS, GENERATION_DEFAULTS, DEFAULT_N_CLOS
from model_loader   import load_tokenizer, load_model, build_pipeline
from prompt_builder import build_messages
from clo_generation      import generate_clos, display_clos


def parse_args():
    
    parser = argparse.ArgumentParser(description="CLO Generator — CLI")

    parser.add_argument(
        "--n_clos", type=int, default=DEFAULT_N_CLOS,
        help=f"Number of CLOs to generate (default: {DEFAULT_N_CLOS})",
    )
    parser.add_argument(
        "--temperature", type=float, default=d["temperature"],
        help=f"Sampling temperature (default: {d['temperature']})",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=d["max_new_tokens"],
        help=f"Max output tokens (default: {d['max_new_tokens']})",
    )
    parser.add_argument(
        "--verbs", nargs="+", default=BLOOMS_VERBS,
        help=f"Bloom's verbs to use (default: {BLOOMS_VERBS})",
    )
    parser.add_argument(
        "--content", type=str, default=None,
        help="Path to a .txt file with course content (reads stdin if omitted)",
    )
    return parser.parse_args()


def load_course_content(path: str | None) -> str:
    """Read course content from a file path, stdin, or a built-in example."""
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()

    if not sys.stdin.isatty():
        return sys.stdin.read()

    # Built-in fallback so the CLI works out of the box without a file
    print("[main] No --content file given. Using built-in example content.\n")
    return (
        "Introduction to neural networks, Perceptron, Activation functions, "
        "Back-propagation, Multi-Layer Perceptron, Convolutional Neural Networks, "
        "CNN Layers (Conv, ReLU, Pooling, FC), Hyperparameters (Stride, Depth, Padding), "
        "Regularization (L1, L2, Dropout, Data Augmentation, Early Stopping), "
        "Transfer Learning and Fine Tuning, CNN Architectures (LeNet, AlexNet, VGG, "
        "Inception, ResNet, DenseNet), Autoencoders, Recurrent Neural Networks, "
        "Vanishing Gradient Problem, LSTMs, GANs, Object Detection (R-CNN, YOLO)."
    )


def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  CLO Generator — CLI")
    print(f"{'='*50}")
    print(f"  CLOs        : {args.n_clos}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Verbs       : {args.verbs}")
    print(f"{'='*50}\n")

    course_content = load_course_content(args.content)

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(MODEL_ID)
    model     = load_model(MODEL_ID)
    pipe      = build_pipeline(model, tokenizer)

    # ── Build prompt with runtime values ──────────────────────────────────────
    messages = build_messages(
        course_content=course_content,
        no_of_clos=args.n_clos,
        blooms_verbs=args.verbs,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    gen_args = {
        **GENERATION_DEFAULTS,
        "temperature"   : args.temperature,
        "max_new_tokens": args.max_tokens,
    }
    result = generate_clos(pipe, messages, gen_args)
    display_clos(result)


if __name__ == "__main__":
    main()