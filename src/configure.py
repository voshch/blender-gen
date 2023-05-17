import json
import os
import numpy
import random
from math import ceil as cl, floor as fl
import click


def draw_samples(range, samples):
    return numpy.random.uniform(*range, size=int(samples or 1)).tolist()


def draw_linspace(range, samples):
    return numpy.linspace(*range, samples, endpoint=True).tolist()


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--mode_internal", default="train", help="train|val create training or validation images")
def main(mode_internal):

    config = None
    with open("/data/input/config/config.json") as f:
        config = json.load(f)

    if mode_internal == "train":
        config["output"]["images"] = config["output"]["size_train"]

    elif mode_internal == "val":
        config["output"]["just_merge"] = 0
        config["output"]["images"] = config["output"]["size_val"]

    config["output"]["just_merge"] = min(
        max(config["output"]["just_merge"], 0), 1)

    pos_dof = isinstance(config["random"]["x_pos"], list) or isinstance(
        config["random"]["y_pos"], list) or isinstance(config["random"]["z_pos"], list)

    os.makedirs("/data/intermediate/config/", exist_ok=True)

    # RENDER

    conf_render = config["render"]

    with open("/data/intermediate/config/render.json", "w") as f:
        json.dump(conf_render, f)

    # TARGETS

    to_produce = config["output"]["images"]
    to_produce *= (1-config["output"]["just_merge"])
    to_produce /= (len(config["input"]["object"]) or 1) + \
        len(config["input"]["distractor"])

    dof_ang = isinstance(config["random"]["inc"], list) + \
        isinstance(config["random"]["azi"], list)
    dof_mat = isinstance(config["random"]["metallic"], list) + \
        isinstance(config["random"]["roughness"], list)

    each = (to_produce / (config["output"]["skew_angle:material"]
            ** dof_mat)) ** (1/max(1, dof_ang + dof_mat))

    targets = dict(
        inc=max(1, cl(each)),
        azi=max(1, cl(each)),
        metallic=max(1, cl(each / config["output"]["skew_angle:material"])),
        roughness=max(1, cl(each / config["output"]["skew_angle:material"]))
    )

    for target in targets:
        if isinstance(config["random"][target], list):
            targets[target] = draw_linspace(
                config["random"][target], targets[target])
        else:
            targets[target] = [config["random"][target]]

    max_size = max(map(lambda x: x["size"], [
                   *config["input"]["object"], * config["input"]["distractor"]]))

    conf_targets = dict(
        object=list(),
        distractor=list(),
        environment=config["input"]["environment"]
    )

    for obj in config["input"]["object"]:
        obj["size"] = (obj["size"] / max_size) ** (1/3)  # 3 dimensions
        conf_targets["object"].append(dict(
            **obj,
            **targets
        ))

    for distractor in config["input"]["distractor"]:
        distractor["size"] /= max_size
        conf_targets["distractor"].append(dict(
            **distractor,
            **targets
        ))

    with open("/data/intermediate/config/targets.json", "w") as f:
        json.dump(conf_targets, f)

    # MERGE

    backgrounds = None
    if "backgrounds" in config["input"]:
        # 360° environment not implemented yet
        backgrounds = config["input"]["backgrounds"]
    else:
        backgrounds = os.listdir("/data/input/backgrounds/static")

    if len(backgrounds) == 0:
        backgrounds = [None]

    conf_merge = []

    dof_pos_x = isinstance(config["random"]["x_pos"], list)
    dof_pos_y = isinstance(config["random"]["y_pos"], list)
    dof_pos_z = isinstance(config["random"]["z_pos"], list)

    dof_distractors = isinstance(config["random"]["distractors"], list)

    no_objects = len(config["input"]["object"])

    for i in range(config["output"]["images"]):
        merge = dict(
            backgrounds=dict(
                name=random.choice(backgrounds)
            ),
            object=[],
            distractor=[]
        )

        for obj in config["input"]["object"]:
            for j in range(obj["multiplicity"] if "multiplicity" in obj else 1):
                merge["object"].append(dict(
                    name=f'{obj["model"]}-{random.choice(targets["inc"])}-{random.choice(targets["azi"])}-{random.choice(targets["metallic"])}-{random.choice(targets["roughness"])}.png',
                    translation=[
                        draw_samples(config["random"]["x_pos"], 1)[
                            0] if dof_pos_x else config["random"]["x_pos"],
                        draw_samples(config["random"]["y_pos"], 1)[
                            0] if dof_pos_y else config["random"]["y_pos"],
                        draw_samples(config["random"]["z_pos"], 1)[
                            0] if dof_pos_z else config["random"]["z_pos"]
                    ]
                ),)
        random.shuffle(merge["object"])

        for j in range(random.randint(*config["random"]["distractors"]) if dof_distractors else config["random"]["distractors"]):
            merge["distractor"].append(dict(
                name=f'{random.choice(config["input"]["distractor"])["model"]}-{random.choice(targets["inc"])}-{random.choice(targets["azi"])}-{random.choice(targets["metallic"])}-{random.choice(targets["roughness"])}.png',
                translation=[
                    draw_samples(config["random"]["x_pos"], 1)[
                        0] if dof_pos_x else config["random"]["x_pos"],
                    draw_samples(config["random"]["y_pos"], 1)[
                        0] if dof_pos_y else config["random"]["y_pos"],
                    draw_samples(config["random"]["z_pos"], 1)[
                        0] if dof_pos_z else config["random"]["z_pos"]
                ]
            ),)

        conf_merge.append(merge)

    with open("/data/intermediate/config/merge.json", "w") as f:
        json.dump(conf_merge, f)

    with open("/data/intermediate/config/postfx.json", "w") as f:
        json.dump(config["postfx"] if "postfx" in config else {}, f)

    total = (len(config["input"]["object"]) + len(config["input"]
             ["distractor"])) * numpy.prod([len(targets[k]) for k in targets])

    print(f"Configured {total} renders for {len(conf_merge)} images")
    print("Breakdown:")
    print(
        f'Objects:    {(len(config["input"]["object"]) + len(config["input"]["distractor"]))}')
    print(f'inc:        {len(targets["inc"])}')
    print(f'azi:        {len(targets["azi"])}')
    # print(f'metallic:   {len(targets["metallic"])}')
    # print(f'roughness:  {len(targets["roughness"])}')
    print("")


if __name__ == "__main__":
    main()
