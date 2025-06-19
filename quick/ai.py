import requests, os, json

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

leading_prompt = f"""You are a coding expert for pulse sequence experiment with package `quick`. Generate a Python function run with the signature `def run(ip="", data_path="", title="", var={{}})`. Generate only the function code concisely with NO comments.

Keyword arguments:
- `ip`: QICK IP address. Use `soccfg, soc = quick.connect(ip)` to connect to the QICK board.
- `data_path`: the directory path to save data.
- `title`: the filename of the data
- `var`: a Python dictionary that inputs relevant parameters. Do NOT modify var.

When user specify values in the instruction, use user's values. If not, use the values in the `var` dictionary, which has the following default values.
```yaml
var: # default variables
    rr: 0             # readout channel
    r: 0              # generator channel for readout pulse
    q: 2              # generator channel for qubit pulse
    r_freq: 5000      # [MHz] readout pulse frequency
    r_power: -30      # [dB] readout pulse power
    r_length: 2       # [us] readout pulse length
    r_phase: 0        # [deg] readout pulse phase
    r_offset: 0       # [us] readout window offset. Unless specified by the user, you should always offset the trigger step by r_offset to compensate for ADC signal processing time.
    r_threshold: 0    # threshold, above which is 1-state
    r_reset: 0        # [us] wait time for qubit reset (active reset)
    r_relax: 1        # [us] readout relax time
    q_freq: 5000      # [MHz] qubit pulse frequency
    q_length: 2       # [us] qubit pulse length
    q_length_2: null  # [us] half pi pulse length
    q_delta: -180     # [MHz] qubit anharmonicity
    q_gain: 1         # [-1, 1] qubit pulse (pi pulse) gain
    q_gain_2: 0.5     # [-1, 1] half pi pulse gain
```

Assume the following imports are available:
```python{imports}```

Always write a Mercator Protocol in YAML string (avoid using {{}}, avoid extra spaces or empty lines), then use `quick.evalStr` to insert variables into Mercator protocol, and use `yaml.safe_load` to convert it into Python dictionary. When possible, omit optional properties and use default values (e.g. `sigma`, `rs`). Be careful about timing. You might need to include Python expression to calculate length and time.

When the task needs sweeping variables, use `for _var in quick.Sweep(var, sweep_config)`.

If the task requires saving data, use `quick.Saver` with `title` and `data_path`, save the following information in `params`:
- `mercator`: the Mercator protocol YAML string
- `var`: the input `var` (Python dictionary)
- `quick_version`: `quick.__version__`

Refer to following documentation for details and examples.
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
    def generate(self, api_key, model="gemini-2.5-pro", silent=False):
        if not silent:
            print("quick.AI Generating...")
        res = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}", {
            "system_instruction": { "parts": [ { "text": self.system_instruction() } ] },
            "contents": [ { "parts": [ { "text": self.prompt } ] } ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": { "code": { "type": "string" } },
                    "required": [ "code" ]
                }
            }
        })
        self.code = json.loads(res["candidates"][0]["content"]["parts"][0]["text"])["code"]
        if not silent:
            print("\n--- quick.AI Generated Code ---")
            print(self.code)
            print("\n--- Token Usage ---")
            print(f"Input: {res['usageMetadata']['promptTokenCount']}\nOutput: {res['usageMetadata']['totalTokenCount'] - res['usageMetadata']['promptTokenCount']}")
        return self
    def run(self, **kwargs):
        template_var = dict(quick.experiment.var)
        template_var.update(kwargs.get("var", {}))
        kwargs["var"] = template_var
        exec(imports + "\n" + self.code, globals())
        return run(**kwargs)
