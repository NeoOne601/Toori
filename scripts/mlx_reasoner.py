"""
mlx_reasoner.py — Gemma 4 e4b on-device reasoning for Toori.

Protocol (unchanged from existing contract):
  stdin:  JSON lines { "prompt": str, "image_base64": str|null,
                       "system": str|null, "max_tokens": int|null }
  stdout: JSON lines { "text": str, "tokens_generated": int,
                       "model": str, "latency_ms": float,
                       "local": bool, "vision_used": bool }
"""
from __future__ import annotations
import base64, json, sys, time, tempfile, os
from pathlib import Path
from typing import Optional

LOCAL_MODEL_PATH = "/Volumes/Apple/AI Model/gemma-4-e4b-it-4bit"
FALLBACK_HF_REPO = "mlx-community/gemma-4-e4b-it-4bit"
DEFAULT_MAX_TOKENS = 512
TEMP = 0.2
TOP_P = 0.9

_MODEL = _TOKENIZER = _LOADED_PATH = None

def _get_model():
    global _MODEL, _TOKENIZER, _LOADED_PATH
    if _MODEL is None:
        import mlx_vlm
        path = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else FALLBACK_HF_REPO
        print(f"[mlx_reasoner] Loading from {path}", file=sys.stderr, flush=True)
        _MODEL, _TOKENIZER, _LOADED_PATH = *mlx_vlm.load(path), path
        print("[mlx_reasoner] Model ready", file=sys.stderr, flush=True)
    return _MODEL, _TOKENIZER, _LOADED_PATH

def _build_prompt(user_text, system, has_image, processor, model):
    parts = []
    if system:
        parts.append(system.strip())
    parts.append(user_text.strip())
    
    # mlx_vlm.apply_chat_template expects a prompt string or list of messages
    # and a config object/dict as the second argument.
    messages = [{"role": "user", "content": "\n".join(parts)}]
    if has_image:
        messages[0]["content"] = [{"type": "image"}] + [{"type": "text", "text": "\n".join(parts)}]
    
    import mlx_vlm
    return mlx_vlm.apply_chat_template(processor, model.config, messages, add_generation_prompt=True)

def _generate(prompt, image_path, max_tokens, model, processor):
    import mlx_vlm
    kwargs = dict(model=model, processor=processor, prompt=prompt,
                  max_tokens=max_tokens, temp=TEMP, top_p=TOP_P, verbose=False)
    if image_path:
        kwargs["image"] = image_path
    
    result = mlx_vlm.generate(**kwargs)
    text = result.text if hasattr(result, "text") else str(result)
    for m in ["<end_of_turn>", "<eos>", "</s>"]:
        text = text.replace(m, "").strip()
    tokens = max(1, len(text.split()) * 4 // 3)
    return text, tokens

def _save_tmp_image(b64: str) -> str:
    import base64, tempfile
    data = base64.b64decode(b64)
    suffix = ".jpg" if data[:3] == b"\xff\xd8\xff" else ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data); tmp.flush(); tmp.close()
    return tmp.name

def _write_error(msg):
    print(f"[mlx_reasoner] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.stdout.write(json.dumps({"text":"","tokens_generated":0,"model":"gemma-4-e4b",
                                  "latency_ms":0.0,"local":True,"error":msg}) + "\n")
    sys.stdout.flush()

def main():
    model, tokenizer, loaded_path = _get_model()
    model_label = Path(loaded_path).name if loaded_path else "gemma-4-e4b"
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw: continue
        tmp = None
        try:
            p = json.loads(raw)
            user_text = str(p.get("prompt","")).strip()
            image_b64 = p.get("image_base64") or None
            system    = p.get("system") or None
            max_tok   = int(p.get("max_tokens") or DEFAULT_MAX_TOKENS)
            if not user_text: _write_error("Empty prompt"); continue
            if image_b64:
                try: tmp = _save_tmp_image(image_b64)
                except Exception as e: print(f"[mlx_reasoner] img err: {e}", file=sys.stderr); tmp = None
            prompt = _build_prompt(user_text, system, bool(image_b64), tokenizer, model)
            t0 = time.perf_counter()
            text, tc = _generate(prompt, tmp, max_tok, model, tokenizer)
            latency = (time.perf_counter() - t0) * 1000.0
            out = {"text":text,"tokens_generated":tc,"model":model_label,
                   "latency_ms":round(latency,1),"local":True,"vision_used":tmp is not None}
            sys.stdout.write(json.dumps(out) + "\n"); sys.stdout.flush()
        except json.JSONDecodeError as e: _write_error(f"JSON: {e}")
        except Exception as e: _write_error(str(e))
        finally:
            if tmp and os.path.exists(tmp):
                try: os.unlink(tmp)
                except OSError: pass

if __name__ == "__main__": main()
