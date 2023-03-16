import util
import cv2 as cv
import numpy as np
import json
import os
import sys
import click
import grequests as requests
import base64
import shapely

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Parameters:
    preview_size = 10
    preview_width = 480
    preview_height = 360
    preview_ext = ".jpg"
    preview_MIME = "image/jpg"


cfg = None
with open("/data/intermediate/config/render.json") as f:
    cfg = json.load(f)

storage = None


def reset():
    global storage
    storage = dict(
        backgrounds=dict(),
        object=dict(),
        distractor=dict()
    )


reset()


def load(target: str, name: str):

    if name is storage[target]:
        return storage[target][name]

    if target == "backgrounds":
        storage[target][name] = cv.resize(
            cv.imread(
                f"/data/intermediate/backgrounds/{name}", cv.IMREAD_UNCHANGED),
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

def get_trf(x: np.float64, y: np.float64, z: np.float64):
    z = z if z > -1+1e-1 else -1+1e-1  # maybe clip to zero instead idk

    trf = np.array([
        [1/(1+z), 0, (x + .5*z/(1+z))*cfg["resolution_x"]],
        [0, 1/(1+z), (y + .5*z/(1+z))*cfg["resolution_y"]]
    ], dtype=np.float32)

    return trf

def transform(img: np.ndarray, trf: np.ndarray):
    return cv.warpAffine(
        img,
        trf,
        (cfg["resolution_x"], cfg["resolution_y"])
    )


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


def merge(backgrounds, obj, distractor=[]):

    trfs = []

    im_obj = load("object", obj["name"])
    im_bg = load("backgrounds", backgrounds["name"]) if backgrounds["name"] != None else np.zeros(
        (cfg["resolution_y"], cfg["resolution_x"], *im_obj.shape[2:]))

    im_distractor = list(map(lambda x: load("distractor", x["name"]), distractor))

    img = np.asarray(im_bg.copy(), dtype=np.float64)

    trf = get_trf(*obj["translation"])
    trfs.append(trf)

    im_obj = transform(im_obj, trf)
    layer(img, im_obj)

    for im_dist, dist in zip(im_distractor, distractor):

        trf = get_trf(*dist["translation"])
        trfs.append(trf)

        layer(img, transform(im_dist, trf))

    return img, trfs

def occlude(clippee: shapely.Polygon, clipper: shapely.Polygon):
    diff = clippee.difference(clipper)
    if isinstance(diff, shapely.MultiPolygon):
        return clippee
    return diff

def trf_vec2(trf, vec2):
    return [(trf @ np.array([*vec, 1])).tolist() for vec in vec2]

def create_preview(img):
    imgdata = cv.imencode(Parameters.preview_ext, cv.resize(img, (Parameters.preview_width, Parameters.preview_height), cv.INTER_AREA))[1]
    imgdata_enc = base64.b64encode(imgdata).decode("utf-8")
    return f"data:{Parameters.preview_MIME};base64,{imgdata_enc}"

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--endpoint", default=None, help="http endpoint for sending current progress")
@click.option("--taskID", default="", help="task ID")
@click.option("--coco-image-root", default="./", help="http endpoint for sending current progress")
@click.option("--mode_internal")
def main(endpoint, taskid, coco_image_root, mode_internal):

    merges = None
    with open("/data/intermediate/config/merge.json") as f:
        merges = json.load(f)

    object_annotations = None
    with open("/data/intermediate/render/renders/object/annotations.json") as f:
        object_annotations = json.load(f)

    distractor_annotations = {}
    try:
        with open("/data/intermediate/render/renders/distractor/annotations.json") as f:
            distractor_annotations = json.load(f)
    except FileNotFoundError:
        pass;

    camera_K = None
    with open("/data/intermediate/render/camera_intrinsic.json") as f:
        camera_K = json.load(f)

    basepath = os.path.join("/data/output/", mode_internal)

    os.makedirs(os.path.join(basepath, "images"), exist_ok=True)
    os.makedirs(os.path.join(basepath, "dota"), exist_ok=True)

    coco_img = []
    coco_label = []

    total = len(merges)
    digits = len(str(total-1))

    if endpoint != None:
        preview0, trfs = merge(merges[-1]["backgrounds"], merges[-1]["object"])
        requests.post(f"{endpoint}/datasetPreview/", json=dict(
            taskId=taskid,
            mode=mode_internal,
            image=create_preview(preview0)
        ))

    print(f"\r{0:0{digits}} / {total}", end="", flush=True)

    for i, conf in enumerate(merges):

        merged, trfs = merge(conf["backgrounds"], conf["object"], conf["distractor"])
        trf_obj = trfs[0]

        id = f"{i:0{digits}}"

        cv.imwrite(os.path.join(basepath, f"images/{id}.png"), merged)

        if (endpoint != None) and (i < Parameters.preview_size - 1):
            requests.post(f"{endpoint}/datasetPreview/", json=dict(
                taskId=taskid,
                mode=mode_internal,
                image=create_preview(merged)
            )).send()

        coco_img.append({
            "id": id,
            "file_name": os.path.join(coco_image_root, f"images/{id}.png"),
            "height": cfg["resolution_x"],
            "width": cfg["resolution_y"],
        })

        annotation = object_annotations[conf["object"]["name"]]

        bbox = [
            trf_obj[0, 0] * annotation["bbox"][0] + trf_obj[0, 2],  # x1
            trf_obj[1, 1] * annotation["bbox"][1] + trf_obj[1, 2],  # x1
            trf_obj[0, 0] * annotation["bbox"][2],  # x2
            trf_obj[1, 1] * annotation["bbox"][3]  # y2
        ]


        segmentation = []

        if annotation["hull"] != None:
            segmentation = shapely.Polygon(trf_vec2(trf_obj, annotation["hull"]))

            for i, distractor in enumerate(conf["distractor"]):
                segmentation = occlude(segmentation, shapely.Polygon(trf_vec2(trfs[i+1], distractor_annotations[distractor["name"]]["hull"])))

            segmentation = list(segmentation.boundary.coords)

        #coco
        coco_label.append({
            "id": id,  # overwrite
            "image_id": id,
            "caption": annotation["label"],
            "category_id": 0,
            "segmentation": [[coord for vec in segmentation for coord in vec]], #flat
            "iscrowd": 0,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "keypoints": [],
            "num_keypoints": 0
        })


        trf_rotated_bbox = trf_vec2(trf_obj, annotation["rotated_bbox"])

        with open(os.path.join(basepath, f"dota/{id}.txt"), "w") as f:
            f.write(f"{', '.join([str(coord) for vec in trf_rotated_bbox for coord in vec])}, {annotation['label']}, 0\n")

        print(f"\r{i+1:0{digits}} / {total}", end="", flush=True)

    print()
    util.saveCOCOlabel(coco_img, coco_label, camera_K, basepath)


if __name__ == "__main__":
    main()
