import asyncio
import aiohttp
from datetime import datetime
import requests
from PIL import Image
import random
import os
from typing import List
from io import BytesIO

OFAN_URL = "https://www.team-eerola.fi/vlrunner/get_runners_ofan.php"
DEFAULT_IMAGE_URL = "https://fanappbucket2022.s3.eu-north-1.amazonaws.com/team_pictures/default_team.jpg"
CACHE_FILE = "cached_runners.txt"

async def latest_modification_later_than(
        photo_url: str,
        latest_run: datetime,
        session: aiohttp.ClientSession
    ) -> str:
    async with session.head(photo_url) as resp:
        last_mod = resp.headers["Last-Modified"]
        dt = datetime.strptime(last_mod, "%a, %d %b %Y %H:%M:%S %Z")
        if dt > latest_run:
            return photo_url


async def get_new(photo_urls: List[str], latest_run: datetime) -> List[str]:
    print(f"making {len(photo_urls)} request to s3 asynchronously")
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        results = await asyncio.gather(*[
            latest_modification_later_than(
                url, latest_run, session) for url in photo_urls])
    return [r for r in results if r is not None]


async def get_runners_async(
        n: int | None = None, 
        start:int | None = None, 
        latest_run: datetime = datetime(2000,1,1)
    ) -> List[str]:

    if os.path.exists(CACHE_FILE):
        print("reading from cache:",CACHE_FILE)
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            res = f.read()
    else:
        print("request to", OFAN_URL)
        res = requests.get(OFAN_URL)
        res = res.text.rstrip()
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(res)
        print("cached to", CACHE_FILE)

    rows = res.split("\n")[1:]
    if n:
        random.shuffle(rows)
    else:
        n = len(rows) - (start if start else 0)
    if start:
        rows = rows[start:]
    rows = rows[:n]
    photo_urls = [i.split(";")[6] for i in rows]
    photo_urls = [j for j in photo_urls if j != "" and j != DEFAULT_IMAGE_URL]
    new_rows = await get_new(photo_urls, latest_run)
    return new_rows


def get_photo(url: str, target_size: int = 3e5, downsample: bool = True) -> Image.Image:
    res = requests.get(url)
    img = Image.open(BytesIO(res.content))
    if not downsample:
        return img

    h, w = img.size
    r = round((h * w / target_size) ** 0.5)
    if r > 1:
        # print(f"{(h, w)}->{(h//r, w//r)}, {r=}, {h//r * w//r} pixels")
        img = img.resize((h // r, w // r))
    return img