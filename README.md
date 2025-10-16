# Mini GPT Code Assistant

This repository provides `mini_gpt_code_v1.py`, a self-contained Python script that implements a miniature GPT-style code assistant. The assistant can train a byte-level BPE tokenizer and decoder-only Transformer from scratch, index a project for retrieval-augmented answers, and run an interactive CLI helper for coding questions without relying on third-party AI APIs.

## Prerequisites

- Python 3.9 or newer
- [PyTorch](https://pytorch.org/get-started/locally/) installed for your platform
- Optional: `requests` library (for whitelisted HTTP documentation fetching)

Install the optional dependency with:

```bash
pip install requests
```

### Fixing Windows launcher errors when installing `requests`

If you see an error similar to:

```
Fatal error in launcher: Unable to create process using "C:\Users\<you>\AppData\Local\Programs\Python\Python313\python.exe" "...\pip.exe" install requests
```

the Python launcher is pointing to an interpreter that no longer exists. Use one of the following approaches to repair your
environment:

1. **Use the interpreter that is actually installed**

   ```powershell
   py -3 -m pip install requests
   ```

   or explicitly target your active Python executable:

   ```powershell
   "C:\Path\To\Your\Python\python.exe" -m pip install requests
   ```

2. **Repair the broken launcher entry**

   - Reinstall or repair Python from https://www.python.org/downloads/ and check the box to add Python to `PATH`.
   - After reinstalling, run `py -0p` to confirm the launcher sees the correct interpreter path.

3. **Clean up stale `pip.exe` shims**

   - Remove any outdated `pip.exe` files from `C:\Users\<you>\AppData\Local\Programs\Python\Python313\Scripts`.
   - Open a new terminal so the refreshed PATH is picked up, then run `py -m ensurepip --upgrade` followed by `py -m pip install requests`.

Once the launcher points to a valid interpreter, rerun `pip install requests` (or `python -m pip install requests`) and the
installation should succeed.

## Basic Usage

All functionality lives in `mini_gpt_code_v1.py`. Run `python mini_gpt_code_v1.py --help` to see every available argument. Below are the most common workflows.

### 1. Prepare a Training Corpus and Train the Model

Use `--train` to build a tokenizer, prepare datasets, and train a small GPT. If you do not provide `--data`, the script can auto-collect code and text from the project directory.

```bash
python mini_gpt_code_v1.py --project_dir . --train --out out
```

You can resume from the most recent checkpoint with:

```bash
python mini_gpt_code_v1.py --project_dir . --resume --out out
```

### 2. Sample from the Model

After training, sample text continuations without retrieval using:

```bash
python mini_gpt_code_v1.py --sample --start "def fibonacci(n):" --max_new_tokens 120 --temperature 0.9
```

### 3. Launch the Assistant

Run the retrieval-augmented CLI assistant that watches your project for changes and optionally fetches live documentation (if `requests` is installed):

```bash
python mini_gpt_code_v1.py --assistant --project_dir . --out out --enable_web true
```

Limit the indexed file extensions or adjust the reindex polling interval if needed:

```bash
python mini_gpt_code_v1.py --assistant --index_extensions ".py,.md" --reindex_interval 3
```

### 4. Additional Options

- `--device auto|cpu|cuda` selects the computation device.
- `--compile` attempts to run the model through `torch.compile` when available.
- `--max_context_tokens` caps the combined retrieval and prompt context size.

## Testing the Script

You can verify that the script compiles by running:

```bash
python -m compileall mini_gpt_code_v1.py
```

## Troubleshooting

- If you omit `--data` and the auto-collected corpus is too large, reduce the project directory or create a curated training file.
- When `requests` is unavailable, the assistant will skip live web fetching and suggest installing it for documentation lookups.
- The assistant never executes arbitrary commands; it only uses safe git inspection commands.

Refer to the inline docstrings in `mini_gpt_code_v1.py` for deeper implementation details.
