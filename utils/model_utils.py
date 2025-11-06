# deimv2_hgnetv2_atto_coco.pth https://drive.google.com/file/d/18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc/view
# deimv2_hgnetv2_femto_coco.pth https://drive.google.com/file/d/16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds/view
# deimv2_hgnetv2_pico_coco.pth https://drive.google.com/file/d/1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT/view
# deimv2_hgnetv2_n_coco.pth https://drive.google.com/file/d/1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K/view
# deimv2_dinov3_s_coco.pth https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view
# deimv2_dinov3_m_coco.pth https://drive.google.com/file/d/1nPKDHrotusQ748O1cQXJfi5wdShq6bKp/view 
# deimv2_dinov3_l_coco.pth https://drive.google.com/file/d/1dRJfVHr9HtpdvaHlnQP460yPVHynMray/view
# deimv2_dinov3_x_coco.pth https://drive.google.com/file/d/1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7/view

import os
import sys
import torch
import torch.nn as nn
import threading
import gdown


# Add DEIMv2 to path
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
deimv2_dir = os.path.join(base_dir, 'DEIMv2')
sys.path.append(os.path.abspath(deimv2_dir))
from engine.core import YAMLConfig

# Mapping from model_name to checkpoint filename and config filename
# Format: model_name -> (checkpoint_filename, config_filename, backbone_type)
MODEL_MAPPING = {
    # DINOv3 models
    "s_coco": ("deimv2_dinov3_s_coco.pth", "deimv2_dinov3_s_coco.yml", "dinov3"),
    "m_coco": ("deimv2_dinov3_m_coco.pth", "deimv2_dinov3_m_coco.yml", "dinov3"),
    "l_coco": ("deimv2_dinov3_l_coco.pth", "deimv2_dinov3_l_coco.yml", "dinov3"),
    "x_coco": ("deimv2_dinov3_x_coco.pth", "deimv2_dinov3_x_coco.yml", "dinov3"),
    # HGNetV2 models
    "atto_coco": ("deimv2_hgnetv2_atto_coco.pth", "deimv2_hgnetv2_atto_coco.yml", "hgnetv2"),
    "femto_coco": ("deimv2_hgnetv2_femto_coco.pth", "deimv2_hgnetv2_femto_coco.yml", "hgnetv2"),
    "pico_coco": ("deimv2_hgnetv2_pico_coco.pth", "deimv2_hgnetv2_pico_coco.yml", "hgnetv2"),
    "n_coco": ("deimv2_hgnetv2_n_coco.pth", "deimv2_hgnetv2_n_coco.yml", "hgnetv2"),

}

# Mapping from checkpoint filename to Google Drive file ID
# Format: checkpoint_filename -> drive_file_id
DRIVE_FILE_IDS = {
    "deimv2_hgnetv2_atto_coco.pth": "18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc",
    "deimv2_hgnetv2_femto_coco.pth": "16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds",
    "deimv2_hgnetv2_pico_coco.pth": "1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT",
    "deimv2_hgnetv2_n_coco.pth": "1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K",
    "deimv2_dinov3_s_coco.pth": "1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL",
    "deimv2_dinov3_m_coco.pth": "1nPKDHrotusQ748O1cQXJfi5wdShq6bKp",
    "deimv2_dinov3_l_coco.pth": "1dRJfVHr9HtpdvaHlnQP460yPVHynMray",
    "deimv2_dinov3_x_coco.pth": "1pTiQaBGt8hwtO0mbYlJ8nE-HGztGafS7",
}

# Lock dictionary to prevent concurrent downloads of the same file
_download_locks = {}
_locks_lock = threading.Lock()


def _get_download_lock(file_path):
    """Get or create a lock for a specific file download."""
    with _locks_lock:
        if file_path not in _download_locks:
            _download_locks[file_path] = threading.Lock()
        return _download_locks[file_path]


def download_model(checkpoint_filename, checkpoint_path):
    """
    Download model from Google Drive if it doesn't exist.
    
    Args:
        checkpoint_filename: Name of the checkpoint file
        checkpoint_path: Full path where the checkpoint should be saved
        
    Raises:
        ImportError: If gdown is not installed
        ValueError: If checkpoint_filename is not in DRIVE_FILE_IDS
        RuntimeError: If download fails
    """
    # Check if gdown is available
    if gdown is None:
        raise ImportError(
            "gdown is required for automatic model download. "
            "Please install it with: pip install gdown"
        )

    # Check if file already exists
    if os.path.exists(checkpoint_path):
        return

    # Check if we have a drive ID for this file
    if checkpoint_filename not in DRIVE_FILE_IDS:
        raise ValueError(
            f"No download URL found for {checkpoint_filename}. "
            f"Available models: {list(DRIVE_FILE_IDS.keys())}"
        )

    # Get or create lock for this specific file
    file_lock = _get_download_lock(checkpoint_path)

    # Acquire lock to prevent concurrent downloads
    with file_lock:
        # Double-check if file exists after acquiring lock
        # (another thread might have downloaded it while we were waiting)
        if os.path.exists(checkpoint_path):
            return

        # Create directory if it doesn't exist
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Construct Google Drive download URL
        file_id = DRIVE_FILE_IDS[checkpoint_filename]
        url = f"https://drive.google.com/uc?id={file_id}"

        # Download the file
        print(f"Downloading {checkpoint_filename} from Google Drive...")
        try:
            gdown.download(url, checkpoint_path, quiet=False)

            # Verify download succeeded
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Download failed: {checkpoint_path} was not created")

            print(f"Successfully downloaded {checkpoint_filename}")
        except Exception as e:
            # Clean up partial download if it exists
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except:
                    pass
            raise RuntimeError(f"Failed to download {checkpoint_filename}: {str(e)}")


def load_model(param):
    """
    Load DEIMv2 model based on model_name parameter.
    
    Args:
        param: InferDeimV2Param object containing model_name and other parameters
        
    Returns:
        model: The loaded model (wrapped in Model class)
        postprocessor: The postprocessor for the model
    """


    if not param.model_weight_file: # use default model
        model_name = param.model_name
    # Check if model_name is in mapping
        if model_name not in MODEL_MAPPING :
            raise ValueError(
                f"Unknown model_name: {model_name}. Available models: {list(MODEL_MAPPING.keys())}")
        # map model_name to checkpoint_filename and config_filename
        checkpoint_filename, config_filename, _ = MODEL_MAPPING[model_name]
        # construct checkpoint path
        checkpoint_path = os.path.join(deimv2_dir,"..",'weights', checkpoint_filename)

        if param.config_file:
            config_path = param.config_file
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            config_path = os.path.join(deimv2_dir, 'configs', 'deimv2', config_filename)
        # Try to download model if it doesn't exist and no custom path is provided
        download_model(checkpoint_filename, checkpoint_path)

    else: # use custom model
        checkpoint_path = param.model_weight_file
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if not param.config_file:
            raise ValueError(
                f"Config file not provided, please provide a config file or use a default model")
        else:
            config_path = param.config_file
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config with resume parameter (checkpoint path)
    # args.resume is always True/checkpoint path based on user's requirement
    cfg = YAMLConfig(config_path, resume=True)

    # Disable HGNetv2 pretrained if using HGNetv2 backbone
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # Load checkpoint (resume is always True/checkpoint path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    # Create Model wrapper class (similar to torch_inf.py)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    # Extract size from config - try eval_spatial_size first, then Resize transform
    size = cfg.yaml_cfg.get('eval_spatial_size')

    if size is None:
        # Try to extract from dataloader transforms
        for loader_key in ['val_dataloader', 'train_dataloader']:
            loader = cfg.yaml_cfg.get(loader_key, {})
            ops = loader.get('dataset', {}).get('transforms', {}).get('ops', [])
            for op in ops:
                if isinstance(op, dict) and op.get('type') == 'Resize':
                    size = op.get('size')
                    if size:
                        break
            if size:
                break

    if size is None:
        raise ValueError(
            "Could not find size in config. Expected 'eval_spatial_size' or 'Resize' transform with 'size' parameter.")

    return model, model.postprocessor, size
