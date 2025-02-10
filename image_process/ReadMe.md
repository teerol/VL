# VL runner image crop

From user uploaded images of runners, resize crop (and remove background).

![Example Image](example/v.png)
![Example Image](example/Figure_234.png)

## Basic Usage

The main file is the `crop_images.py`. In beginning the target image size and wanted scaling of the face are defined. 

**NOTE:** Background removal requires additional library (using neural nets on CPU) and its SLOW.

`get_runners_ansync.py` provides utilities to anyncronously check if there is any updated images loaded. The eventual loading of the image is still syncronous.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/teemueer/VL
    cd VL/image_process
    ```

2. Create and activete virtual environment with dependecies 
    ```sh
    ./venv.cmd
    ```

## Running `crop_images.py`

To run the `crop_images.py` script, use the following options:
```sh
  -h, --help            show this help message and exit
  --show                show images, blocks the execution
  --save                save images to output_path
  --latest_run LATEST_RUN
                        Takes only images updated after latest run! Optional parameter, latest run datetime, format "YYYY-MM-DD HH:MM:SS". If not provided, the
                        latest run will be read from "latest_run.txt" or set to beginning of time
  --output_path OUTPUT_PATH
                        output path to save images. Crop and background removed images will be saved to
                        'output_path/crop' and 'output_path/bg' respectively
  --rem_bg              remove background, NOTE: This is a lot slower and results vary.
```

### Example
```sh
python crop_images.py --save --output_path images/
```