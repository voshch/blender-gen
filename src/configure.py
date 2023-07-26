import json
import os
import numpy
import random
from math import ceil as cl, floor as fl, pi
import click

from util import Mode

def draw_samples(range, samples):
    return numpy.random.uniform(*range, size=int(samples or 1)).tolist()


def draw_linspace(range, samples):
    return numpy.linspace(*range, samples, endpoint=True).tolist()


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--mode_internal", type=click.Choice([Mode.Train, Mode.Val]), default=Mode.Train, help="train|val create training or validation images")
def main(mode_internal):

    config = None
    with open("/data/input/config/config.json") as f:
        config = json.load(f)

    match mode_internal:
        case Mode.Train:
            config["output"]["images"] = config["output"]["size_train"]

        case Mode.Val:
            config["output"]["just_merge"] = 0
            config["output"]["images"] = config["output"]["size_val"]

        case _:
            raise ValueError(f"unknown mode {mode_internal}")

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

    each = to_produce ** (1/max(1, dof_ang))

    d2r = pi/180  # deg2rad
    config["random"]["inc"] = (
        d2r * numpy.array(config["random"]["inc"])).tolist()
    config["random"]["azi"] = (
        d2r * numpy.array(config["random"]["azi"])).tolist()

    targets = dict(
        inc=max(1, cl(each)),
        azi=max(1, cl(each))
    )

    for target in targets:
        if isinstance(config["random"][target], list):
            targets[target] = draw_linspace(
                config["random"][target], targets[target])
        else:
            targets[target] = [config["random"][target]]

    max_size = max([x["size"] if "size" in x else 0 for x in
                    [*config["input"]["object"], * config["input"]["distractor"]]
                    ]) or 1

    conf_targets = dict(
        object=list(),
        distractor=list(),
        environment=config["input"]["environment"]
    )

    for obj in config["input"]["object"]:
        obj["size"] = (obj["size"] / max_size) ** (1/3) if "size" in obj else 1
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
                    name=f'{obj["model"]}-{random.choice(targets["inc"])}-{random.choice(targets["azi"])}.png',
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
                name=f'{random.choice(config["input"]["distractor"])["model"]}-{random.choice(targets["inc"])}-{random.choice(targets["azi"])}.png',
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

    total = (len(config["input"]["object"]) + len(config["input"]
             ["distractor"])) * numpy.prod([len(targets[k]) for k in targets])

    print(f"Configured {total} renders for {len(conf_merge)} images")
    print("Breakdown:")
    print(
        f'Objects:    {(len(config["input"]["object"]) + len(config["input"]["distractor"]))}')
    print(f'inc:        {len(targets["inc"])}')
    print(f'azi:        {len(targets["azi"])}')
    print("")


if __name__ == "__main__":
    main()
