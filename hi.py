from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access variables
example_name = os.getenv("EXAMPLE_NAME")
models_dir = os.getenv("MODELS_DIR")
cfg_file = os.getenv("CFG_FILE")
port = os.getenv("PORT")

print("EXAMPLE_NAME:", example_name)
print("MODELS_DIR:", models_dir)
print("CFG_FILE:", cfg_file)
print("PORT:", port)
