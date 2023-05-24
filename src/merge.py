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
from enum import Enum
import itertools

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Parameters:
    preview_size = 10
    preview_width = 480
    preview_height = 360
    preview_ext = ".jpg"
    preview_MIME = "image/jpg"


class Image(Enum):
    OBJECT = 1
    DISTRACTOR = 2
    BACKGROUND = 3


class Categories:
    categories = dict()

    def __init__(self):
        self.generator = itertools.count()

    def get(self, name):
        if name not in self.categories:
            self.categories[name] = self.generator.__next__()

        return self.categories[name]


categories = Categories()

cfg = None
with open("/data/intermediate/config/render.json") as f:
    cfg = json.load(f)

storage = None


def reset():
    global storage
    storage = dict()
    storage[Image.OBJECT] = dict()
    storage[Image.DISTRACTOR] = dict()
    storage[Image.BACKGROUND] = dict()


reset()


def load(target: str, name: str):

    if name in storage[target]:
        return storage[target][name]

    if target == Image.BACKGROUND:
        storage[target][name] = cv.resize(
            cv.imread(
                f"/data/intermediate/backgrounds/{name}", cv.IMREAD_UNCHANGED),
            (cfg["resolution_x"], cfg["resolution_y"]),
            interpolation=cv.INTER_AREA
        )

    elif target == Image.OBJECT:
        storage[target][name] = cv.imread(
            f"/data/intermediate/render/renders/object/{name}", cv.IMREAD_UNCHANGED)

    elif target == Image.DISTRACTOR:
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


def merge(backgrounds, objects=[], distractor=[]):

    images_obj = list(map(lambda x: load(Image.OBJECT, x["name"]), objects))

    im_bg = load(Image.BACKGROUND, backgrounds["name"]) if backgrounds["name"] != None else np.zeros(
        (cfg["resolution_y"], cfg["resolution_x"], *images_obj[0].shape[2:]))

    images_distractor = list(
        map(lambda x: load(Image.DISTRACTOR, x["name"]), distractor))

    img = np.asarray(im_bg.copy(), dtype=np.float64)

    transformations_obj = []
    for im_obj, obj in zip(images_obj, objects):
        trf = get_trf(*obj["translation"])
        transformations_obj.append(trf)

        layer(img, transform(im_obj, trf))

    transformations_dist = []
    for im_dist, dist in zip(images_distractor, distractor):

        trf = get_trf(*dist["translation"])
        transformations_dist.append(trf)

        layer(img, transform(im_dist, trf))

    return img, transformations_obj, transformations_dist


def occlude(clippee: shapely.MultiPolygon, clipper: shapely.MultiPolygon):

    try:
        clipped = clippee.difference(clipper)

        if isinstance(clipped, shapely.MultiPolygon):
            return clipped

        if isinstance(clipped, shapely.GeometryCollection):
            return shapely.MultiPolygon(list(filter(lambda x: isinstance(x, shapely.Polygon), clipped.geoms)))

        return shapely.MultiPolygon([clipped])

    except shapely.geos.TopologicalError:
        return clippee


def trf_vec2(trf, vec2):
    return [(trf @ np.array([*vec, 1])).tolist() for vec in vec2]


def create_preview(img):
    imgdata = cv.imencode(Parameters.preview_ext, cv.resize(
        img, (Parameters.preview_width, Parameters.preview_height), cv.INTER_AREA))[1]
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

    object_annotations = {}
    with open("/data/intermediate/render/renders/object/annotations.json") as f:
        object_annotations = json.load(f)

    distractor_annotations = {}
    try:
        with open("/data/intermediate/render/renders/distractor/annotations.json") as f:
            distractor_annotations = json.load(f)
    except FileNotFoundError:
        pass

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
        preview0 = merge(merges[-1]["backgrounds"], merges[-1]["object"])[0]
        requests.post(f"{endpoint}/datasetPreview/", json=dict(
            taskId=taskid,
            mode=mode_internal,
            image=create_preview(preview0)
        ))

    warnings = ""
    print(f"\r{0:0{digits}} / {total}", end="", flush=True)

    for i, conf in enumerate(merges):

        merged, trfs_obj, trfs_dist = merge(
            conf["backgrounds"], conf["object"], conf["distractor"])

        id = f"{i:0{digits}}"

        cv.imwrite(os.path.join(basepath, f"images/{id}.png"), merged)
        coco_img.append({
            "id": id,
            "file_name": os.path.join(coco_image_root, f"images/{id}.png"),
            "height": cfg["resolution_x"],
            "width": cfg["resolution_y"],
        })

        dota_file = open(os.path.join(basepath, f"dota/{id}.txt"), "w")

        if (endpoint != None) and (i < Parameters.preview_size - 1):
            requests.post(f"{endpoint}/datasetPreview/", json=dict(
                taskId=taskid,
                mode=mode_internal,
                image=create_preview(merged)
            )).send()

        for n, trf_obj in enumerate(trfs_obj):

            annotation = object_annotations[conf["object"][n]["name"]]

            bbox = [
                trf_obj[0, 0] * annotation["bbox"][0] + trf_obj[0, 2],  # x1
                trf_obj[1, 1] * annotation["bbox"][1] + trf_obj[1, 2],  # x1
                trf_obj[0, 0] * annotation["bbox"][2],  # x2
                trf_obj[1, 1] * annotation["bbox"][3]  # y2
            ]

            segmentation = []

            if annotation["hull"] != None:
                segmentation = shapely.MultiPolygon(
                    [shapely.Polygon(trf_vec2(trf_obj, segment)) for segment in annotation["hull"]])

                for q, obj in list(enumerate(conf["object"]))[n+1:]:
                    shape = shapely.MultiPolygon([shapely.Polygon(trf_vec2(
                        trfs_obj[q], segment)) for segment in object_annotations[obj["name"]]["hull"]])
                    segmentation = occlude(segmentation, shape)

                for q, distractor in enumerate(conf["distractor"]):
                    shape = shapely.MultiPolygon([shapely.Polygon(trf_vec2(
                        trfs_dist[q], segment)) for segment in distractor_annotations[distractor["name"]]["hull"]])
                    segmentation = occlude(segmentation, shape)

                try:
                    segmentation = [list(segment.boundary.coords)
                                    for segment in segmentation.geoms]
                except NotImplementedError:
                    segmentation = [[]]
                    warnings += f"object {n} in image {i} produced invalid segmentation\n"

            # coco
            coco_label.append({
                "id": id,  # overwrite
                "image_id": id,
                "caption": annotation["label"],
                "category_id": categories.get(annotation["label"]),
                # flat
                "segmentation": [[coord for vec in segment for coord in vec] for segment in segmentation],
                "iscrowd": 0,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "keypoints": [],
                "num_keypoints": 0
            })

            dota_file.write(
                f"{', '.join([str(coord) for vec in trf_vec2(trf_obj, annotation['rotated_bbox']) for coord in vec])}, {annotation['label']}, 0\n")

        dota_file.close()

        print(f"\r{i+1:0{digits}} / {total}", end="", flush=True)

    if (warnings != ""):
        print(f"\n[WARNINGS]\n{warnings}\n")

    util.saveCOCOlabel(coco_img, coco_label, camera_K,
                       basepath, categories.categories)
    
    print("\n")


if __name__ == "__main__":
    main()
