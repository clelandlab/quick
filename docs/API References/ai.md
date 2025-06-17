# AI

Artificial Intelligence for quick.

**This functionality requires Internet connection. Currently it has zero knowledge about `quick.experiment` or `quick.auto`.**

> Only AI *(class)* is exposed as `quick.AI`.

## 🔵AI

```python
ai = quick.AI(prompt)
```

The AI class to generate code. On construction, it will load documentation of quick from Github.

> By default, the generated code will always be a Python function `run(ip="", data_path="", title="", var={})`.

**Parameters**:

- `prompt` (str) prompt for the AI to generate code. Be specific and it is recommended to include the structure of your `var` if you have non-standard variables.

### 🔵AI.system_instruction

```python
s = ai.system_instruction()
```

**Returns**:

- `s` (str) the assembled system instruction. This is the instruction that will be used in `ai.generate`.

### 🔵AI.generate

```python
ai = ai.generate(api_key, model="gemini-2.5-pro-preview-06-05", silent=False)
```

Generate code using the AI model.

**Parameters**:

- `api_key` (str) API key for the AI model.
- `model="gemini-2.5-pro-preview-06-05"` (str) AI model to use.
- `silent=False` (bool) whether to avoid printing the generated code. By default, it will print the generated code and the token usage.

**Returns**:

- `ai` (AI) the AI instance with the generated code. The generated code will be stored in `ai.code`.

### 🔵AI.run

```python
r = ai.run(**kwargs)
```

Run the generated code with the provided keyword arguments.

**Parameters**: should match the arguments of the generated function (following by default)

- `ip=""` (str) IP address of the QICK board.
- `data_path=""` (str) directory to save data.
- `title=""` (str) title of the data.
- `var={}` (dict) experiment variables. Default variables will be added if missing.

**Returns**:

- `r` the return of the generated function.
