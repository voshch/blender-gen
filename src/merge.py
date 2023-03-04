import cv2 as cv
import numpy as np
import json
import os
import sys
import click
import grequests as requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import util

cfg = None
with open("/data/intermediate/config/render.json") as f:
    cfg = json.load(f)

storage = None


def reset():
    global storage
    storage = {
        "backgrounds": {},
        "object": {},
        "distractor": {}
    }


reset()


def load(target: str, name: str):

    if name is storage[target]:
        return storage[target][name]

    if target == "backgrounds":
        storage[target][name] = cv.resize(
            cv.imread(f"/data/intermediate/backgrounds/{name}", cv.IMREAD_UNCHANGED),
            (cfg["resolution_x"], cfg["resolution_y"]),
            interpolation=cv.INTER_AREA
        )

    elif target == "object":
        storage[target][name] = cv.imread(
            f"/data/intermediate/render/renders/object/{name}", cv.IMREAD_UNCHANGED)

    elif target == "distractor":
        storage[target][name] = cv.imread(
            f"/data/intermediate/render/renders/distractor/{name}", cv.IMREAD_UNCHANGED)

    return storage[target][name]


def transform(img: np.ndarray, x: np.float64, y: np.float64, z: np.float64):
    z = z if z > -1+1e-1 else -1+1e-1  # maybe clip to zero instead idk

    trf = np.array([
        [1/(1+z), 0, (x + .5*z/(1+z))*cfg["resolution_x"]],
        [0, 1/(1+z), (y + .5*z/(1+z))*cfg["resolution_y"]]
    ], dtype=np.float32)

    return cv.warpAffine(
        img,
        trf,
        (cfg["resolution_x"], cfg["resolution_y"])
    ), trf


def layer(img: np.ndarray, overlay: np.ndarray):
    alpha = overlay[..., -1]/255.
    alphas = np.dstack([alpha] * img.shape[2])
    img *= 1-alphas
    
    desired_shape = img.shape[2]
    
    new_overlay = overlay
    overlay_shape_offset = desired_shape - overlay.shape[2]
    
    if overlay_shape_offset < 0:
        new_overlay = overlay[..., :(desired_shape - overlay.shape[2])]
    
    new_alphas = alphas 
    alpha_shape_offset = desired_shape - alphas.shape[2]

    if alpha_shape_offset < 0:
        new_alphas = alphas[..., :(desired_shape - alphas.shape[2])]

    img += new_alphas * new_overlay
    return img


def merge(backgrounds, obj, distractor):
    im_bg = load("backgrounds", backgrounds["name"])
    im_obj = load("object", obj["name"])
    im_distractor = list(
        map(lambda x: load("distractor", x["name"]), distractor))

    img = np.asarray(im_bg.copy(), dtype=np.float64)

    im_obj, trf = transform(im_obj, *obj["translation"])
    layer(img, im_obj)

    for im_dist, dist in zip(im_distractor, distractor):
        layer(img, transform(im_dist, *dist["translation"])[0])

    return img, trf


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--endpoint", default=None, help="http endpoint for sending current progress")
@click.option("--taskID", default="", help="task ID")
@click.option("--coco-image-root", default="/data/output/", help="http endpoint for sending current progress")
@click.option("--mode_internal")
def main(endpoint, taskid, coco_image_root, mode_internal):

    merges = None
    with open("/data/intermediate/config/merge.json") as f:
        merges = json.load(f)

    annotations = None
    with open("/data/intermediate/render/annotations.json") as f:
        annotations = json.load(f)

    camera_K = None
    with open("/data/intermediate/render/camera_intrinsic.json") as f:
        camera_K = json.load(f)

    basepath = os.path.join("/data/output/", mode_internal)

    os.makedirs(os.path.join(basepath, "images"), exist_ok=True)

    coco_img = []
    coco_label = []

    total = len(merges)
    digits = len(str(total))

    print(f"\r{0:0{digits}} / {total}", end="", flush=True)

    for i, conf in enumerate(merges):

        merged, trf = merge(conf["backgrounds"], conf["object"], conf["distractor"])

        id = f"{i:0{digits}}"

        cv.imwrite(os.path.join(basepath, f"images/{id}.png"), merged)

        coco_img.append({
            "id": id,
            "file_name": os.path.join(coco_image_root, mode_internal, f"images/{id}.png"),
            "height": cfg["resolution_x"],
            "width": cfg["resolution_y"],
        })

        annotation = annotations[conf["object"]["name"]]

        annotation["bbox"] = [
            trf[0, 0] * annotation["bbox"][0] + trf[0, 2],  # x1
            trf[1, 1] * annotation["bbox"][1] + trf[1, 2],  # x1
            trf[0, 0] * annotation["bbox"][2],  # x2
            trf[1, 1] * annotation["bbox"][3]  # y2
        ]

        coco_label.append({
            **annotation,
            "id": id, #overwrite
            "image_id": id,
            "category_id": 0,
            "segmentation": [],
            "iscrowd": 0,
            "area": annotation["bbox"][2] * annotation["bbox"][3],
            "num_keypoints": len(annotation["keypoints"])
        })

        print(f"\r{i+1:0{digits}} / {total}", end="", flush=True)

        if not endpoint == None:
            requests.post(f"{endpoint}/output", data=dict(
                id=taskid,
                progress=i+1,
                total=total
            )).send()

    print()
    util.saveCOCOlabel(coco_img, coco_label, camera_K, basepath)


if __name__ == "__main__":
    main()
