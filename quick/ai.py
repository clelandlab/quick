import requests, os

# execution imports
import quick, yaml, scipy, IPython
import numpy as np
import matplotlib.pyplot as plt

imports = """
import quick, yaml, scipy, IPython
import numpy as np
import matplotlib.pyplot as plt
"""

def get(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def post(url, data):
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()

def get_tag():
    try: # test if tag exists
        get(f"https://raw.githubusercontent.com/clelandlab/quick/refs/tags/{quick.__version__}/docs/index.md")
        return f"tags/{quick.__version__}"
    except:
        return "heads/main"

tag = ""

leading_prompt = f"""Use the following documentation to complete the task by generating a Python function `run`. Write concise code. Do not include any comments in the code. The generated function should take the following keyword arguments:
- `ip=""`: QICK IP address. Use `soccfg, soc = quick.connect(ip)` to connect to the QICK board.
- `data_path=""`: the path to the directory to save data.
- `title=""`: the filename of the data
- `var={{}}`: a python dictionary that input relevant paramemters for the task. Do NOT modify var.

Only generate code for the `run` function. Assume the following imports are available:
```python{imports}```

Always write a Mercator Protocol in YAML string (avoid using {{}}), then use `quick.evalStr` to insert variables into Mercator protocol, and use `yaml.safe_load` to convert it into Python dictionary. When the task needs sweeping variables, use `for _var in quick.Sweep(var, sweep_config)`. If the task requires saving data, use `quick.Saver`. Refer to documentation for details and examples.
"""

class AI:
    def __init__(self, prompt):
        global tag
        if tag == "":
            tag = get_tag()
        self.tag = tag
        self.prompt = prompt
        self.docs = []
        for doc in ['Tutorials/mercator.md', 'API References/mercator.md', 'API References/helper.md']:
            res = get(f"https://raw.githubusercontent.com/clelandlab/quick/refs/{tag}/docs/{doc}")
            self.docs.append(res.replace("🟢", "").replace("🔵", "").replace("🟡", ""))
        self.code = ""
    def system_instruction(self):
        return leading_prompt + "\n\n\n" + "\n\n\n".join(self.docs)
    def generate(self, api_key, model="gemini-2.5-pro-preview-06-05", silent=False):
        if not silent:
            print("quick.AI Generating...")
        res = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}", {
            "system_instruction": { "parts": [ { "text": self.system_instruction() } ] },
            "contents": [ { "parts": [ { "text": self.prompt } ] } ],
            "generationConfig": { "temperature": 0 }
        })
        self.code = res["candidates"][0]["content"]["parts"][0]["text"]
        if not silent:
            print("\n--- quick.AI Generated Code ---")
            print(self.code)
            print("\n--- Token Usage ---")
            print(f"Input: {res['usageMetadata']['promptTokenCount']}\nOutput: {res['usageMetadata']['totalTokenCount'] - res['usageMetadata']['promptTokenCount']}")
        return self
    def run(self, **kwargs):
        exec(imports + "\n" + self.code, globals())
        return run(**kwargs)
