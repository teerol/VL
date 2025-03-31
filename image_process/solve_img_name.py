from pathlib import Path
import os
import requests
from dataclasses import dataclass
from get_runners_async import CACHE_FILE, OFAN_URL, DEFAULT_IMAGE_URL

@dataclass
class Runner:
    name: str
    club: str
    imgurl: str
    original: Path | None
    crop: Path | None
    bgpath: Path | None

def solve_img_names(imgurl: str) -> tuple[str, str, str]:
    imgpath = Path(imgurl)
    fol = "-".join(imgpath.parts[-4: -1])
    original = f"originals/{fol}-{imgpath.name}"
    crop = f"crop/{fol}-{imgpath.name}"
    bg = f"bg/{fol}-{imgpath.stem}.png"
    return original, crop, bg

def get_runners():
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
    runners = []
    for row in rows:
        # id;seuralong;seurashort;Nimi;Description;Motto;imageURL
        row = row.split(";")
        photo_url = row[6]
        if (photo_url == "" or 
            photo_url == DEFAULT_IMAGE_URL or 
            photo_url.lower().endswith(".heic")
        ):
            continue    

        original, crop, bg = solve_img_names(photo_url)
        runner = Runner(
            name=row[3],
            club=row[2].replace(".png",""),
            imgurl=row[6],
            original=original,
            crop=crop,
            bgpath=bg,
        )
        runners.append(runner)
    return runners
    
def write_index(runners: list[Runner], index_path: Path) -> None:
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("name;club;imgurl;original;crop;bgpath\n")
        for runner in runners:
            f.write(f"{runner.name};{runner.club};{runner.imgurl};{runner.original};{runner.crop};{runner.bgpath}\n")
    print(f"wrote {len(runners)} runners to {index_path}")


if __name__ == "__main__":
    import sys
    import argparse
    argparser = argparse.ArgumentParser(description="Get runners from OFAN")
    argparser.add_argument("--index_path", type=str, help="Path to index file", default="runners_index.csv", required=False)
    args = argparser.parse_args(sys.argv[1:])
    runners = get_runners()
    index_path = Path(args.index_path)
    if not index_path.parent.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
    write_index(runners, index_path)