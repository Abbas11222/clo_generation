
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import MODEL_ID


def load_tokenizer(model_id: str = MODEL_ID):
    """Load and return the tokenizer for the given model ID."""
    print(f"[ModelLoader] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


def load_model(model_id: str = MODEL_ID):
    """Load and return the causal LM model with automatic device mapping."""
    print(f"[ModelLoader] Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    return model


def build_pipeline(model, tokenizer):
    """Wrap model + tokenizer into a text-generation pipeline."""
    print("[ModelLoader] Building text-generation pipeline ...")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe