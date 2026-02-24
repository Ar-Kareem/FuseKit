import os
from pathlib import Path
import yaml

# ---- Default base dirs ----
HOME = Path.home()
CWD  = Path.cwd()
DEFAULT_ROOT = CWD / "fusekit_data"
DEFAULT_MODELS = "/storage/cmarnold/models"

# Provide a default models location when FUSEKIT_MODELS is unset.
os.environ.setdefault("FUSEKIT_MODELS", DEFAULT_MODELS)

# Environment variable override
ROOT = Path(os.getenv("FUSEKIT_ROOT", DEFAULT_ROOT))

APIKEYS_DIR = Path(os.getenv("FUSEKIT_APIKEYS", ROOT / "apikeys"))
MODELS_DIR  = Path(os.getenv("FUSEKIT_MODELS",  ROOT / "models"))
DATASETS_DIR = Path(os.getenv("FUSEKIT_DATASETS", ROOT / "datasets"))
ARTIFACTS_DIR = Path(os.getenv("FUSEKIT_ARTIFACTS", ROOT / "."))
CONFIG_FILE  = Path(os.getenv("FUSEKIT_CONFIG", ROOT / "config.yml"))

def needs_init() -> bool:
    return not (APIKEYS_DIR.exists() and MODELS_DIR.exists())

if needs_init() and os.isatty(0) and os.getenv("FUSEKIT_NO_PROMPT") != "1":
    print("FuseKit not initialized. Run `fusekit init`")

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        try:
            cfg = yaml.safe_load(f) or {}
            ROOT = Path(cfg.get("root", ROOT))
        except Exception:
            cfg = {}
else:
    cfg = {}

# Set the project directory based on the current file's location
PROJECT_DIR = Path(__file__).resolve().parent.parent
CWD_DIR = Path(os.getcwd())

DEFAULT_DATASETS_DIR = PROJECT_DIR / 'Datasets'

results = ARTIFACTS_DIR / 'results'
adapters = ARTIFACTS_DIR / 'adapters'

# TODO: Move to MapQA-Survey
survey_responses = PROJECT_DIR / 'survey-responses'
class SurveyResponses:
    root = CWD_DIR / 'survey-responses'
    military = root / 'MilitaryAccuracy'
    natural_world = root / 'NaturalWorldAccuracy'
    urban = root / 'UrbanAccuracy'

class VerifyPath:
    def __getattribute__(self, name: str):
        # Fetch the attribute normally
        value = object.__getattribute__(self, name)

        # Skip dunders / internals quickly
        if name.startswith("__"):
            return value

        # Validate only Path objects (dirs or files)
        if isinstance(value, Path):
            if not value.exists():
                raise FileNotFoundError(
                    f"Model path not found: {value}\n"
                    "set config.yml with correct directory"
                )
        return value

class _APIKeys(VerifyPath):
    openai = APIKEYS_DIR / 'openai.apikey'
    openai_org = APIKEYS_DIR / 'openai.org'
    claude = APIKEYS_DIR / 'claude.apikey'
    claude_org = APIKEYS_DIR / 'claude.org'
    gemini = APIKEYS_DIR / 'gemini.apikey'
    
class _ModelPath(VerifyPath):
    llama2_7b = MODELS_DIR/ 'llama2' / '7B'
    llama2_13b = MODELS_DIR / 'llama2' / '13B'
    llama2_70b = MODELS_DIR / 'llama2' / '70B'

    llava_next_7b_vicuna = MODELS_DIR / 'llava-next' / '7B-Vicuna'
    llava_next_7b_mistral = MODELS_DIR / 'llava-next' / '7B-Mistral'
    llava_next_13b_vicuna = MODELS_DIR / 'llava-next' / '13B-Vicuna'
    llava_next_34b = MODELS_DIR / 'llava-next' / '34B'
    llava_next_72b = MODELS_DIR / 'llava-next' / '72B'
    llava_next_110b = MODELS_DIR / 'llava-next' / '110B'
    
    pixtral_12b = MODELS_DIR / 'pixtral' / '12B'

    qwen2_2b = MODELS_DIR / 'qwen2' / '2B-Instruct'
    qwen2_7b = MODELS_DIR / 'qwen2' / '7B-Instruct'
    
    phi3_5_vision = MODELS_DIR / 'phi3' / 'Vision-Instruct'

    llama3_8b = MODELS_DIR / 'llama3' / '8B'
    llama3_11b_vision = MODELS_DIR/ 'llama3' / '11B-Vision-Instruct'
    llama3_90b_vision = MODELS_DIR / 'llama3' / '90B-Vision-Instruct'
    
APIKeys = _APIKeys()
ModelPath = _ModelPath()

class DatasetPath:
    commonsenseqa = DEFAULT_DATASETS_DIR / 'CommonsenseQA'
