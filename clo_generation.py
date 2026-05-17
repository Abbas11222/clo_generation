
from config import GENERATION_DEFAULTS


def generate_clos(pipe, messages: list[dict], generation_args: dict = None) -> str:
   
    args = generation_args if generation_args is not None else GENERATION_DEFAULTS

    print("[Generator] Running inference ...")
    output = pipe(messages, **args)

    result: str = output[0]["generated_text"]
    return result
