from __future__ import annotations

import json
import click
import os
from typing import *
import cv2 as cv
import numpy as np

import util

rng = np.random.default_rng()


class Node:
    inputs: List[Node]
    outputs: List[Node]
    value: Iterable[cv.Mat]

    def input(self, node: Node) -> None:
        self.inputs.append(node)
        node.outputs.append(self)

    def output(self, node: Node) -> None:
        self.outputs.append(node)
        node.inputs.append(self)

    def propagate(self) -> None:
        for output in self.outputs:
            output.propagate()

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.value = []

    def configure(self, *args):
        pass

    def __repr__(self) -> str:
        return f"Node({repr(type(self).__name__)})"


class Plugin(Node):
    configuration: dict = {"passthrough": True}

    def propagate(self) -> None:
        inputs = (mat.copy() for inp in self.inputs for mat in inp.value)
        self.value = inputs if self.configuration["passthrough"] else self.processor(
            inputs)
        super().propagate()

    def processor(self, inputs: Iterable[cv.Mat]) -> Iterable[cv.Mat]:
        return inputs

    def configure(self, configuration: dict = {}) -> None:
        print(f"configured {self} with {configuration}")
        self.configuration = {**self.configuration,
                              **configuration,
                              "passthrough": configuration == {}
                              }

    def __init__(self):
        super().__init__()
        self.configure()


class LinearBlur(Plugin):
    configuration: dict = dict(
        angle=0,
        length=0
    )

    def processor(self, inputs: Iterable[cv.Mat]) -> Iterable[cv.Mat]:

        angle = self.configuration["angle"]
        length = self.configuration["length"]

        def blur(inp: cv.Mat):
            # https://stackoverflow.com/questions/40817634/opencv-how-to-add-artificial-smudge-motion-blur-effects-to-a-whole-image

            # opencv raises error for length < 6 because ???
            padded_length = max(6, length)
            kernel_size = 2*padded_length + 1
            kernel = np.zeros((kernel_size, kernel_size, inp.shape[-1]))
            kernel = cv.ellipse(kernel,
                                (padded_length, padded_length),
                                (length, 0),
                                angle,
                                0, 360,
                                (1, 1, 1),
                                thickness=-1)
            kernel /= kernel[:, :, 0].sum()

            cv.filter2D(inp, -1, kernel, inp)
            return inp

        return map(blur, inputs)


class GaussianBlur(Plugin):
    configuration: dict = dict(
        radius=1
    )

    def processor(self, inputs: Iterable[cv.Mat]) -> Iterable[cv.Mat]:

        radius = self.configuration["radius"]
        radius += 1 - radius % 2  # odd number

        def blur(inp: cv.Mat):
            cv.GaussianBlur(inp, (3*radius, 3*radius), radius, inp)
            return inp

        return map(blur, inputs)

# http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
# https://homes.psd.uchicago.edu/~ejmartin/pix/20d/tests/noise/


class ShotNoise(Plugin):
    configuration: dict = dict(
        # relative amount, could be calculated as 1/sqrt(k * brightness)
        amount=0
    )

    def processor(self, inputs: Iterable[cv.Mat]) -> Iterable[cv.Mat]:

        amount = self.configuration["amount"]

        def shot(inp: cv.Mat):
            spots = rng.uniform(0, 1, inp.shape) < amount
            inp[spots] = 255
            return inp

        return map(shot, inputs)


class BlackNoise(Plugin):
    configuration: dict = dict(
        # #salt/#pixels = scale/sqrt(#photons) to follow a poisson dist
        mean=0,
        stdev=1
    )

    def processor(self, inputs: Iterable[cv.Mat]) -> Iterable[cv.Mat]:

        mean = self.configuration["mean"]
        stdev = self.configuration["stdev"]

        def black(inp: cv.Mat):
            inp += np.round(rng.normal(mean, stdev, inp.shape)
                            ).astype(inp.dtype)
            inp = np.clip(inp, 0, 255)
            return inp

        return map(black, inputs)


class QuantizationNoise(Plugin):
    configuration: dict = dict(
        baseline=0,
        bits=32
    )

    def processor(self, inputs: Iterable[cv.Mat]) -> Iterable[cv.Mat]:

        baseline = self.configuration["baseline"]
        bits = self.configuration["bits"]

        high = 2 ** bits - 1
        conversion = high/(255-baseline)

        def quant(inp: cv.Mat):
            dtype = inp.dtype
            inp = np.clip(inp, baseline, 255) - baseline
            inp = np.floor(conversion * inp) / conversion
            inp = (inp+baseline).astype(dtype)
            return inp

        return map(quant, inputs)


class InputNode(Node):
    def feed(self, value: List[cv.Mat]):
        self.value = value
        self.propagate()


class OutputNode(Node):
    def propagate(self) -> None:
        self.value = (mat.copy() for inp in self.inputs for mat in inp.value)


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--mode_internal")
@click.option("--taskID", default="manual")
def main(mode_internal, taskid):

    chain = dict(
        input=InputNode(),
        shot=ShotNoise(),
        gaussian=GaussianBlur(),
        motion=LinearBlur(),
        black=BlackNoise(),
        quant=QuantizationNoise(),
        output=OutputNode()
    )

    chain["input"].output(chain["shot"])
    chain["shot"].output(chain["gaussian"])
    chain["gaussian"].output(chain["motion"])
    chain["motion"].output(chain["black"])
    chain["black"].output(chain["quant"])
    chain["quant"].output(chain["output"])

    config: dict = dict()
    with open("/data/input/config/postfx.json") as f:
        config = json.load(f)

    for key in chain:
        chain[key].configure(config[key] if key in config else {})

    srcpath = os.path.join("/data/output/", mode_internal, "images/")
    dstpath = os.path.join(srcpath, "../postfx", taskid)

    os.makedirs(os.path.join(dstpath), exist_ok=True)

    files = [os.path.relpath(os.path.join(root,f), srcpath) for root, dirs, filenames in os.walk(srcpath) for f in filenames]

    for dir in (os.path.relpath(os.path.join(root,d), srcpath) for root, dirs, filenames in os.walk(srcpath, topdown=False) for d in dirs):
        os.makedirs(os.path.join(dstpath, dir), exist_ok=True)

    total = len(files)
    digits = len(str(total-1))

    warnings = ""
    print(f"\r{0:0{digits}} / {total}", end="", flush=True)

    for i, filename in enumerate(files):

        try:
            img = cv.imread(os.path.join(srcpath, filename),
                            cv.IMREAD_UNCHANGED)
            chain["input"].feed([img])
            cv.imwrite(os.path.join(dstpath, filename),
                       list(chain["output"].value)[0])

        except Exception as e:
            warnings += f"Encountered exception while processing file {filename}:\n {e}\n"

        finally:
            print(f"\r{i+1:0{digits}} / {total}", end="", flush=True)

    if (warnings != ""):
        print(f"\n[WARNINGS]\n{warnings}\n")

    print("")


if __name__ == "__main__":
    main()
