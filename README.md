<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_deim_v2</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_deim_v2">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_deim_v2">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_deim_v2/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_deim_v2.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

DEIMv2 is an evolution of the DEIM framework while leveraging the rich features from DINOv3. Our method is designed with various model sizes, from an ultra-light version up to S, M, L, and X, to be adaptable for a wide range of scenarios. Across these variants, DEIMv2 achieves state-of-the-art performance, with the S-sized ("s_coco") model notably surpassing 50 AP on the challenging COCO benchmark.

![object detection](https://raw.githubusercontent.com/Ikomia-hub/infer_deim_v2/main/images/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow
```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_deim_v2", auto_connect=True)

# Run on your image
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_city_2.jpg?raw=true")

# Inpect your result
display(algo.get_image_with_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 's_coco': Name of the DEIMv2 pre-trained model. Other model available:
    - atto_coco
    - femto_coco
    - pico_coco
    - n_coco
    - m_coco
    - l_coco
    - x_coco

- **conf_thres** (float) default '0.5': Box threshold for the prediction [0,1].
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU.


**To load a custom model weights fine-tuned with the _train_d_fine_ algorithm:**
The following file can be found in the train output directory
- **model_weight_file** (str, *optional*): Path to model weights file .pth. 
- **config_file** (str, *optional*): Path to config file .yaml.

**Parameters** should be in **strings format**  when added to the dictionary

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_deim_v2", auto_connect=True)

algo.set_parameters({
    "model_name": "n_coco",
    "conf_thres": "0.5",
    "cuda": "True"
})

# Run on your image
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_city_2.jpg?raw=true")

# Inpect your result
display(algo.get_image_with_graphics())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_deim_v2", auto_connect=True)

# Run on your image
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_city_2.jpg?raw=true")

# Inpect your result
display(algo.get_image_with_graphics())

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```