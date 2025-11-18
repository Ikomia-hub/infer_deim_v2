import copy
import os
import numpy as np

import torch
import yaml
import torchvision.transforms as T
from PIL import Image
from ikomia import core, dataprocess, utils
from ikomia.core import CWorkflowTaskParam
from infer_deim_v2.utils.model_utils import load_model
from infer_deim_v2.utils.coco_labels import COCO_CLASSES


class InferDeimV2Param(CWorkflowTaskParam):

    def __init__(self):
        CWorkflowTaskParam.__init__(self)
        self.model_name = "s_coco"
        self.model_weight_file = ""
        self.cuda = torch.cuda.is_available()
        self.conf_thres = 0.45
        self.config_file = ""
        self.update = False

    def set_values(self, params):
        self.model_name = str(params["model_name"])
        self.model_weight_file = str(params["model_weight_file"])
        self.cuda = utils.strtobool(params["cuda"])
        self.conf_thres = float(params["conf_thres"])
        self.config_file = str(params["config_file"])
        self.update = True

    def get_values(self):
        return {
            "model_name": str(self.model_name),
            "model_weight_file": str(self.model_weight_file),
            "cuda": str(self.cuda),
            "conf_thres": str(self.conf_thres),
            "config_file": str(self.config_file),
            "update": str(self.update)
        }

# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------


class InferDeimV2(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)

        # Create parameters object
        if param is None:
            self.set_param_object(InferDeimV2Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model = None
        self.postprocessor = None
        self.cfg = None
        param_obj = self.get_param_object()
        self.device = torch.device(
            "cuda" if param_obj.cuda and torch.cuda.is_available() else "cpu")
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.size = [640, 640]
        self.labels = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def set_class_names(self, param):
        if param.model_weight_file:
            if not param.config_file:
                raise ValueError(
                    "The 'config_file' is required when using a custom model file.")
            else:
                # load class names from file .txt
                with open(param.config_file, 'r') as f:
                    cfg = yaml.unsafe_load(f)
                    self.labels = cfg['class_names']
                    print(f"Loaded class names: {self.labels}")
        else:
            self.labels = COCO_CLASSES

        # Set class names
        self.set_names(self.labels)

    def run(self):
        self.begin_task_run()
        param = self.get_param_object()

        input_image = self.get_input(0)
        src_image = input_image.get_image()

        # Load model if not loaded or if parameters updated
        if self.model is None or param.update:
            # Update device if CUDA availability changed
            self.device = torch.device(
                "cuda" if param.cuda and torch.cuda.is_available() else "cpu")
            self.model, self.postprocessor, self.size = load_model(param)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.set_class_names(param)

            param.update = False

        # Convert numpy array to PIL Image and ensure RGB format (3 channels)
        if len(src_image.shape) == 3 and src_image.shape[2] == 3:
            src_image = Image.fromarray(src_image).convert('RGB')
        elif len(src_image.shape) == 3 and src_image.shape[2] == 4:
            # Handle RGBA images - convert to RGB
            src_image = Image.fromarray(src_image).convert('RGB')
        else:
            # Handle grayscale or other formats - convert to RGB
            src_image = Image.fromarray(src_image).convert('RGB')

        # Get original image dimensions
        orig_w, orig_h = src_image.size

        # Determine if using DINOv3 backbone (vit_backbone)
        # DINOv3 models: s_coco, m_coco, l_coco, x_coco
        dinov3_models = ["s_coco", "m_coco", "l_coco", "x_coco"]
        vit_backbone = param.model_name in dinov3_models

        # Create transforms
        size = (self.size)
        transforms = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if vit_backbone else T.Lambda(lambda x: x)
        ])

        # Prepare input tensor
        im_data = transforms(src_image).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([[orig_w, orig_h]]).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(im_data, orig_size)

        # Extract results (output is a list of dicts)
        labels, boxes, scores = output
        # Filter by confidence threshold
        keep_mask = scores >= param.conf_thres
        labels = labels[keep_mask]
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]

        # Clamp boxes to image dimensions
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(0, float(orig_w))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(0, float(orig_h))

        # Move to CPU for processing
        labels = labels.detach().cpu()
        boxes = boxes.detach().cpu()
        scores = scores.detach().cpu()

        # Add objects to output
        for idx, (label, box, score) in enumerate(zip(labels, boxes, scores)):
            x1, y1, x2, y2 = box.tolist()
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            self.add_object(
                idx,
                int(label.item()),
                float(score.item()),
                float(x1),
                float(y1),
                float(width),
                float(height)
            )

        self.emit_step_progress()
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDeimV2Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_deim_v2"
        self.info.short_description = "Infer DEIMv2: Real-Time Object Detection Meets DINOv3"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, " \
                            "Xuanlong and Shen, Xi"
        self.info.article = "Real-Time Object Detection Meets DINOv3"
        self.info.journal = "arXiv:2509.20787v2"
        self.info.year = 2025
        self.info.license = "Apache 2.0"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.15.0"

        # Python compatibility
        self.info.min_python_version = "3.9.0"

        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2509.20787"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_deim_v2"
        self.info.original_repository = "https://github.com/Intellindust-AI-Lab/DEIMv2"

        # Keywords used for search
        self.info.keywords = "Object Detection,DINOv3,DEIMv2,COCO"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create algorithm object
        return InferDeimV2(self.info.name, param)
