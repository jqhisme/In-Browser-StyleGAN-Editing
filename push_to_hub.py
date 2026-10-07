from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv()
api = HfApi()

repo = api.create_repo(
    repo_id="e4e-ffhq-onnx",
    repo_type="model",
    exist_ok=True
)

api.upload_folder(
    folder_path = "onnx-export",
    repo_id=repo.repo_id,
    repo_type="model"
)