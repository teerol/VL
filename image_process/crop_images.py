import os
import requests
import argparse
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from PIL.ImageOps import expand

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from get_runners_async import get_runners_async, get_photo

CACHE_DIR = Path("cached")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
FACE_DECTECTOR = CACHE_DIR / "face_detector.tflite"
latest_run_file = CACHE_DIR / "latest_run.txt"

# rembg_model_name = "u2net_human_seg"
rembg_model_name = "birefnet-portrait"

# target metrics for images
# width, height, width_scale, height_scale
img_stats = 300, 280, 1.9, 0.6 # mediapipe setup
img_stats = 300, 280, 2.0, 0.7 # mediapipe setup

# target size when loading from s3
TARGET_LOAD_SIZE = 3e5 # pixels

async def load_and_crop(
        latest_run: str | None,
        output_path: str | None,
        photo_path: str| None,
    ) -> None:
    # to check that are there any new ones
    # read latest run from cache
    if latest_run:
        latest_run = datetime.strptime(latest_run, "%Y-%m-%d %H:%M:%S")
    else:
        latest_run = datetime(2000,1,1)
        if os.path.exists(latest_run_file):
            with open(latest_run_file) as f:
                latest_run = datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
        print("latest run:", latest_run)

    if photo_path:
        # load photos from local folder
        photo_g = [
            os.path.join(root, file) for root, _, files in os.walk(photo_path) for file in files
        ]
    else:
        photo_g = await get_runners_async(latest_run=latest_run)

    with open("latest_run.txt", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if output_path:
        result_folder = Path(output_path)
        (result_folder / "originals").mkdir(parents=True, exist_ok=True)
        (result_folder / "crop").mkdir(parents=True, exist_ok=True)
        (result_folder / "bg").mkdir(parents=True, exist_ok=True)

    # faceCascade = get_haarcascade()
    detector = get_face_detector(confidence=0.5)

    if args.rem_bg:
        session = new_session(model_name=rembg_model_name)

    nface = []
    for i, imgurl in enumerate(photo_g):
        print(f"{i}/{len(photo_g)}, {imgurl}")
        if imgurl.endswith(".heic"):
            print("HEIC FILE! NOT SUPPORTED")
            continue

        if photo_path:
            # load from local folder
            img_o = Image.open(imgurl).convert("RGB")
            imgurl = imgurl.replace(photo_path, "")
        else:
            # s3
            img_o = get_photo(imgurl, TARGET_LOAD_SIZE, downsample=True)

        img, face = detect_with_rotate_mp(img_o, detector, 0.5)
        if not face:
            print("NOT FOUND FACE", imgurl)
            nface.append(imgurl)
            cropped_image = Image.new("RGB", (img_stats[1], img_stats[0]), (0, 0, 0))
            padded_image = Image.new("RGB", (img_stats[1], img_stats[0]), (0, 0, 0))
            if output_path:
                save_images(
                result_folder, imgurl, img_o, cropped_image, padded_image, None,
                )
            continue

        padded_image = resize_and_pad(img, face, *img_stats)

        cropped_image = resize_and_crop(img, face, *img_stats)

        if cropped_image is None:
            print("face too big!", imgurl)
            cropped_image = Image.new("RGB", (img_stats[1], img_stats[0]), (0, 0, 0))

        if args.rem_bg:
            bg_removed = remove(padded_image, session=session)

        if output_path:
           save_images(
               result_folder, imgurl, img_o, cropped_image, padded_image, bg_removed if args.rem_bg else None
            )
        
    with open(CACHE_DIR / "no_face.txt", "w") as f:
        f.write("\n".join(nface))


def save_images(
        result_folder: Path,
        imgurl: str,
        original_image: Image.Image,
        cropped_image: Image.Image,
        padded_image: Image.Image,
        bg_removed: Image.Image | None
    ) -> None:
    imgpath = Path(imgurl)
    fol = "-".join(imgpath.parts[-4: -1]) + "-"
    original_image.save(str(result_folder / "originals" / (fol + imgpath.name)))
    # print(str(result_folder / "crop" / (fol + imgpath.name)))
    if imgpath.suffix == ".jfif":
        # save as jpg, rename to jfif
        asjpg = result_folder / "crop" / (fol + (imgpath.stem + ".jpg"))
        # cv2.imwrite(str(asjpg), cropped_image)
        cropped_image.save(str(asjpg))
        asjpg.rename(result_folder / "crop" / (fol + (imgpath.stem + ".jfif")))
    else:
        # cv2.imwrite(str(result_folder / "crop" / (fol + imgpath.name)), cropped_image)
        cropped_image.save(str(result_folder / "crop" / (fol + imgpath.name)))
        if args.rem_bg and bg_removed is not None:
            bg_removed.save(str(result_folder / "bg" / (fol + imgpath.stem + ".png")))
        else:
            padded_image.save(str(result_folder / "bg" / (fol + imgpath.stem + ".png")))



def get_face_detector(confidence: float = 0.8) -> vision.FaceDetector:
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    if not os.path.exists(FACE_DECTECTOR):
        print("downloading face detector...")
        response = requests.get(url)
        with open(FACE_DECTECTOR, 'wb') as file:
            file.write(response.content)

    model_path = Path(FACE_DECTECTOR)
    with open(model_path, 'rb') as file:
        model_data = file.read()
    base_options = python.BaseOptions(
        model_asset_buffer=model_data,
    )
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=confidence,
        min_suppression_threshold=0.3,
    )
    return vision.FaceDetector.create_from_options(options)

def is_rotation_correct(detection, tolerance_degrees: int=15) -> bool:
    l_eye, r_eye, nose = detection.keypoints[:3]

    dx = r_eye.x - l_eye.x
    dy = r_eye.y - l_eye.y
    nose_y = nose.y

    eye_angle = np.degrees(np.arctan2(dy, dx))

    max_eye_y = max(l_eye.y, r_eye.y)

    if abs(eye_angle) <= tolerance_degrees:
        if nose_y < max_eye_y:
            return False
        return True
    elif (90 - tolerance_degrees) <= eye_angle <= (90 + tolerance_degrees):
        return False
    elif (-90 - tolerance_degrees) <= eye_angle <= (-90 + tolerance_degrees):
        return False
    
    return False    



def detect_with_rotate_mp(
        img: Image.Image, detector: str, confidence: float=0.8
    ) -> tuple[Image.Image | None, tuple[int] | None]:
    best_img, best_box, best_score = None, None, 0
    work_img = img.copy()
    for _ in range(4):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, 
            data=np.asarray(work_img, dtype=np.uint8),
        )

        detection_result = detector.detect(mp_image)
        for detection in detection_result.detections:

            score = detection.categories[0].score
            if score > best_score:# and is_rotation_correct(detection):
                bbox = detection.bounding_box
                best_img = work_img.copy()
                best_box = (
                        int(bbox.origin_x), 
                        int(bbox.origin_y), 
                        int(bbox.width), 
                        int(bbox.height)
                    )
                best_score = detection.categories[0].score
        
        if best_score > 0.9 and is_rotation_correct(detection):
            # shortcut
            break
        work_img = work_img.rotate(angle=270, expand=True) # counter clockwise 90 is more likely

    if best_img is not None and best_score > confidence:
        return best_img, best_box
    
    return None, None


def resize_and_pad(
        img: Image.Image,
        face: tuple[int],
        target_height: int,
        target_width: int,
        width_scale: float,
        height_scale: float,
    ) -> Image.Image:
    w, h = img.size
    x, y, fw, fh = face
    s = target_height / target_width
    dw = int(fw * width_scale)
    dh = int(s * dw)
    dx = int(x + fw / 2 * (1 - width_scale))
    dy = int(y - fh * height_scale)
    if dx + dw > w or dy + dh > h or dx < 0 or dy < 0:
        img = expand(img, fill=0, border=(
            max(-dx, 0),                       # left
            max(-dy, 0) + max(dy + dh - h, 0), # top
            max(dx + dw - w, 0),               # right
            0,                                 # bottom
        ))
        dx = max(dx, 0)
        dy = max(dy, 0)

    return img.resize((target_width, target_height), box=(dx, dy, dx + dw, dy + dh))

def resize_and_crop(
        img: Image.Image,
        face: tuple[int],
        target_height: int,
        target_width: int,
        width_scale: float,
        height_scale: float,
    ) -> Image.Image | None:
    w,h = img.size
    x, y, fw, fh = face
    #cv2.rectangle(img, (x, y), (x+fw, y+fh), (255, 0, 0), 2)
    # cv2_imshow(img)
    s = target_height / target_width
    dw = int(fw * width_scale)
    dh = int(s * dw)
    dx = int(x + fw / 2 * (1 - width_scale))
    dy = int(y - fh * height_scale)
    #print(dx,dy,dw,dh)
    #cv2.rectangle(img,(dx,dy),(dx+dw,dy+dh),(0, 200, 255),2)
    #cv2_imshow(img)
    if dx + dw > w or dy + dh > h or dx < 0 or dy < 0:
        if width_scale > 1.0:
            width_scale -= 0.04
        height_scale -= 0.02
        if height_scale < 0:
            return None
        print(f"retrying with {width_scale=} {height_scale=}")
        return resize_and_crop(img, face, target_height, target_width, width_scale, height_scale)

    # img_crop = img[dy: dy + dh , dx: dx + dw]
    # return cv2.resize(img_crop, (target_width, target_height))
    return img.resize((target_width, target_height), box=(dx, dy, dx + dw, dy + dh))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest_run", type=str, default=None, help='Takes only images updated after latest run! Optional parameter, latest run datetime, format "YYYY-MM-DD HH:MM:SS". If not provided, the latest run will be read from "latest_run.txt" or set to beginning of time')
    parser.add_argument("--output_path", type=str, default=None, help="output path to save images. Original, crop and background removed images will be saved to 'output_path/originals', 'output_path/crop' and 'output_path/bg' respectively")
    parser.add_argument("--rem_bg", action="store_true", help="remove background, NOTE: This is a lot slower and results vary")
    parser.add_argument("--originals_folder", type=str, default=None, help="if given load images from here instead of downloading from s3")

    args = parser.parse_args()

    if args.rem_bg:
        import subprocess
        import sys
        # check if rembg is installed
        try:
            from rembg import remove, new_session
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "rembg[cpu]"])
        from rembg import remove, new_session

    asyncio.run(
        load_and_crop(args.latest_run, args.output_path, args.originals_folder)
    )