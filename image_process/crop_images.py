import os
import requests
import cv2
import argparse
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import datetime
from pathlib import Path

from get_runners_async import get_runners_async, get_photo

CACHE_DIR = Path("cached")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
FACE_CASCADE = CACHE_DIR / "haarcascade_frontalface_default.xml"
latest_run_file = CACHE_DIR / "latest_run.txt"

rembg_model_name = "u2net_human_seg" # birefnet-portrait, sam

# target metrics for images
# width, height, width_scale, height_scale
img_stats = 300, 280, 1.9, 0.4 # new setup

# target size when loading from s3
TARGET_LOAD_SIZE = 3e5 # pixels

async def load_and_crop(show: bool, latest_run: str, output_path) -> None:
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

    photo_g = await get_runners_async(latest_run=latest_run)

    with open("latest_run.txt", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if output_path:
        result_folder = Path(output_path)
        (result_folder / "crop").mkdir(parents=True, exist_ok=True)
        (result_folder / "bg").mkdir(parents=True, exist_ok=True)

    faceCascade = get_haarcascade()

    if args.rem_bg:
        session = new_session(model_name=rembg_model_name)

    nface = []
    for i, imgurl in enumerate(photo_g):
        print(f"{i}/{len(photo_g)}, {imgurl}")
        if imgurl.endswith(".heic"):
            print("HEIC FILE! NOT SUPPORTED")
            continue

        img_o = get_photo(imgurl, TARGET_LOAD_SIZE, downsample=True)
        # convert to cv2 format
        img = cv2.cvtColor(np.array(img_o), cv2.COLOR_RGB2BGR)

        img, face = detect_with_rotate(img, faceCascade, size_limit=.05)
        if not face:
            print("NOT FOUND FACE", imgurl)
            nface.append(imgurl)
            continue

        padded_image = resize_and_pad(img, face, *img_stats)

        cropped_image = resize_and_crop(img, face, *img_stats)

        if len(cropped_image) == 0:
            print("face too big!", imgurl)
            cropped_image = np.zeros_like(padded_image)

        if args.rem_bg:
            bg_removed = remove(padded_image, session=session)

        if show:
            org = np.array(img_o.resize(img_stats[1::-1]))[...,:3][...,::-1]
            cv2_imshow(
                np.concatenate((org, cropped_image, padded_image), axis=1) if not args.rem_bg
                else np.concatenate((org, cropped_image, padded_image, bg_removed[:,:,:-1]), axis=1),
            title="org(not in scale), cropped, padded, background removed", xticks=[], yticks=[])

        if output_path:
           save_images(result_folder, imgurl, cropped_image, bg_removed if args.rem_bg else None)
        
    with open(CACHE_DIR / "no_face.txt", "w") as f:
        f.write("\n".join(nface))


def cv2_imshow(img, **kwargs):
    plt.imshow(img[...,::-1])
    for cmd, val in kwargs.items():
        plt_func = getattr(plt, cmd)
        plt_func(val)
    plt.show()
    # plt.pause(1)

def save_images(result_folder: Path, imgurl: str, cropped_image: np.ndarray, bg_removed: np.ndarray | None) -> None:
    imgpath = Path(imgurl)
    fol = "-".join(imgpath.parts[-4: -1]) + "-"
    print(str(result_folder / "crop" / (fol + imgpath.name)))
    if imgpath.suffix == ".jfif":
        # save as jpg, rename to jfif
        asjpg = result_folder / "crop" / (fol + (imgpath.stem + ".jpg"))
        cv2.imwrite(str(asjpg), cropped_image)
        asjpg.rename(result_folder / "crop" / (fol + (imgpath.stem + ".jfif")))
    else:
        cv2.imwrite(str(result_folder / "crop" / (fol + imgpath.name)), cropped_image)
        if args.rem_bg and bg_removed is not None:
            cv2.imwrite(str(result_folder / "bg" / (fol + (imgpath.stem + ".png"))), bg_removed)


def detect_with_rotate(
        img: np.ndarray,
        faceCascade: cv2.CascadeClassifier,
        size_limit: float
    ) -> Tuple[np.ndarray | None, Tuple[int] | None]:
    for _ in range(3):
        face = detect_face(img, faceCascade, size_limit)
        if face:
            return img, face
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print("rotating")
    return None, None


def detect_face(
        img: np.ndarray,
        faceCascade: cv2.CascadeClassifier,
        size_limit: float,
    ) -> Tuple[int] | None:
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect with confidence
    faces, _, c = faceCascade.detectMultiScale3(
        gray,
        scaleFactor=1.15, # 1.1-1.2
        minNeighbors=4, # 4 or 5
        outputRejectLevels=True)
    if len(faces) > 0:
        # sort by confidence and return the first with size limit applied
        sort_i = np.argsort(-c)
        faces = faces[sort_i]
        for x, y, fw, fh in faces:
            #print(fw/img.shape[0],fh/img.shape[1])
            # relative size of face should be large enough
            if fw / img.shape[0] > size_limit and fh / img.shape[1] > size_limit:
                return (x, y, fw, fh)


def resize_and_pad(
        img: np.ndarray,
        face: Tuple[int],
        target_height: int,
        target_width: int,
        width_scale: float,
        height_scale: float,
    ):
    h, w = img.shape[:2]
    x, y, fw, fh = face
    s = target_height / target_width
    dw = int(fw * width_scale)
    dh = int(s * dw)
    dx = int(x + fw / 2 * (1 - width_scale))
    dy = int(y - fh * height_scale)
    if dx + dw > w or dy + dh > h or dx < 0 or dy < 0:
        #print(h,w)
        #print(dx+dw,dx,dy+dw,w)
        BLACK = [0, 0, 0]
        img_pad = cv2.copyMakeBorder(
            img,
            top=max(-dy, 0) + max(dy + dh - h, 0),
            bottom=0,
            left=max(-dx, 0),
            right=max(dx + dw - w, 0),
            borderType=cv2.BORDER_CONSTANT,
            value=BLACK
        )

        img_pad = img_pad[max(0, dy): dy + dh if dy + dh < h else -1 , max(dx, 0): dx + dw if dx + dw < w else -1]
    else:
        img_pad = img[dy: dy + dh , dx: dx + dw]
    return cv2.resize(img_pad, (target_width, target_height))

def resize_and_crop(
        img: np.ndarray,
        face: Tuple[int],
        target_height: int,
        target_width: int,
        width_scale: float,
        height_scale: float,
    ):
    h, w = img.shape[:2]
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
            return []
        print(f"retrying with {width_scale=} {height_scale=}")
        return resize_and_crop(img, face, target_height, target_width, width_scale, height_scale)

    img_crop = img[dy: dy + dh , dx: dx + dw]
    return cv2.resize(img_crop, (target_width, target_height))

def get_haarcascade() -> cv2.CascadeClassifier:
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    if not os.path.exists(FACE_CASCADE):
        print("downloading haarcascade...")
        response = requests.get(url)
        with open(FACE_CASCADE, 'wb') as file:
            file.write(response.content)
    return cv2.CascadeClassifier(FACE_CASCADE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="show images, blocks the execution")
    parser.add_argument("--latest_run", type=str, default=None, help='Takes only images updated after latest run! Optional parameter, latest run datetime, format "YYYY-MM-DD HH:MM:SS". If not provided, the latest run will be read from "latest_run.txt" or set to beginning of time')
    parser.add_argument("--output_path", type=str, default=None, help="output path to save images. Crop and background removed images will be saved to 'output_path/crop' and 'output_path/bg' respectively")
    parser.add_argument("--rem_bg", action="store_true", help="remove background, NOTE: This is a lot slower and results vary")

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
        load_and_crop(args.show, args.latest_run, args.output_path)
    )