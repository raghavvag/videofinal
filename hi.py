from dotenv import load_dotenv
import os

# Environment configuration loader
load_dotenv()

# Configuration variables retrieval
ENV_VARS = {
    "app_name": os.getenv("EXAMPLE_NAME"),
    "model_directory": os.getenv("MODELS_DIR"),
    "config_file": os.getenv("CFG_FILE"),
    "server_port": os.getenv("PORT")
}

# Display configuration information
print("EXAMPLE_NAME:", ENV_VARS["app_name"])
print("MODELS_DIR:", ENV_VARS["model_directory"])
print("CFG_FILE:", ENV_VARS["config_file"])
print("PORT:", ENV_VARS["server_port"])
