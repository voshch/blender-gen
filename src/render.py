#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# to install packages with PIP into the blender python:
# e.g. /PATH/TO/BLENDER/python/bin$ /python3.7m -m pip install pandas

import traceback

import bpy
import bpy_extras
import os
import sys
import random
import math
import numpy as np
import json
import argparse
import colorsys
import shutil
import glob
from mathutils import Vector, Matrix
import datetime

import shapely
import shapely.ops

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util

log = util.Log()

config = None
with open("/data/intermediate/config/render.json") as f:
    config = json.load(f)


class Target:

    type = ""
    model_path = ""

    def __init__(self, config=None):

        if not config:
            config = {}

        if "inc" not in config:
            config["inc"] = []
        config["inc"] = config["inc"] or [0]

        if "azi" not in config:
            config["azi"] = []
        config["azi"] = config["azi"] or [0]

        if "size" not in config:
            config["size"] = 1

        self.size = config["size"]
        self.model = config["model"]

        self.config = config

    def configs(self):  # lazy iterate over all combinations
        fields = ["inc", "azi"]
        indices = [0 for _ in fields]
        limits = [len(self.config[field]) for field in fields]

        while True:
            yield [self.config[fields[i]][indices[i]] for i in range(0, len(fields))]
            for i in range(len(fields)-1, -1, -1):
                indices[i] += 1
                if indices[i] == limits[i]:
                    if i == 0:
                        return None
                    indices[i] = 0
                else:
                    break

# maybe more useful in the future?


class Object(Target):
    model_path = "/data/input/models/"
    type = "object"

    def __init__(self, config=None):
        super().__init__(config)

        self.config["label"] = config["label"]


class Distractor(Target):
    model_path = "/data/input/models/"
    type = "distractor"

Targets =  dict(
    object=Object,
    distractor=Distractor
)

# def _print(*args, **kwargs):
#     ...
# print = _print

def autoscale(obj, cam=bpy.data.objects['Camera']):
    corners = np.array(obj.bound_box)
    long = np.max(np.abs(corners).max(axis=0))
    scale = 1/np.linalg.norm(project_by_object_utils(cam, Vector(3*[long])))
    log.print(f"{obj.name} max {long} scaled by {scale}")
    obj.scale = (scale, scale, scale)

def importPLYobject(filepath, conf_obj):
    """import PLY object from path and scale it."""

    if conf_obj.model in bpy.data.objects:
        return bpy.data.objects[conf_obj.model]

    bpy.ops.import_mesh.ply(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    obj.name = conf_obj.model
    
    autoscale(obj)
    obj.scale = Vector(conf_obj.size * np.array(obj.scale))

    # add vertex color to PLY object
    obj.select_set(True)
    mat = bpy.data.materials.new(f'Material-{conf_obj.model}')
    obj.active_material = mat
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    vcol = nodes.new(type="ShaderNodeVertexColor")
    vcol.layer_name = "Col"

    mat_links.new(vcol.outputs['Color'], bsdf.inputs['Base Color'])

    # save object material inputs
    # config["metallic"].append(bsdf.inputs['Metallic'].default_value)
    # config["roughness"].append(bsdf.inputs['Roughness'].default_value)

    return obj


def importOBJobject(filepath, conf_obj):
    """import an *.OBJ file to Blender"""

    if conf_obj.model in bpy.data.objects:
        return bpy.data.objects[conf_obj.model]

    bpy.ops.import_scene.obj(filepath=filepath, axis_forward='Y', axis_up='Z')
    # print("importing model with axis_forward=Y, axis_up=Z")

    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()  # join multiple elements into one eleme

    # get BSDF material node
    obj = bpy.context.selected_objects[0]
    obj.name = conf_obj.model

    autoscale(obj)
    obj.scale = Vector(conf_obj.size * np.array(obj.scale))

    mat = obj.active_material
    mat_links = mat.node_tree.links
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    texture = nodes.new(type="ShaderNodeTexImage")
    # mat_links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])

    # save object material inputs
    # config["metallic"].append(bsdf.inputs['Metallic'].default_value)
    # config["roughness"].append(bsdf.inputs['Roughness'].default_value)

    return obj

def importFBXObject(filepath, conf_obj):
    """import an *.FBX file to Blender"""

    if conf_obj.model in bpy.data.objects:
        return bpy.data.objects[conf_obj.model]

    bpy.ops.import_scene.fbx(filepath=filepath, axis_forward='Y', axis_up='Z')
    # print("importing model with axis_forward=Y, axis_up=Z")

    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()  # join multiple elements into one eleme

    # get BSDF material node
    obj = bpy.context.selected_objects[0]
    obj.name = conf_obj.model

    autoscale(obj)
    obj.scale = Vector(conf_obj.size * np.array(obj.scale))

    mat = obj.active_material
    mat_links = mat.node_tree.links
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    texture = nodes.new(type="ShaderNodeTexImage")
    # mat_links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])

    # save object material inputs
    # config["metallic"].append(bsdf.inputs['Metallic'].default_value)
    # config["roughness"].append(bsdf.inputs['Roughness'].default_value)

    return obj

def project_by_object_utils(cam, point):
    """returns normalized (x, y) image coordinates in OpenCV frame for a given blender world point."""
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )

    # convert y coordinate to opencv coordinate system!
    return Vector((co_2d.x, 1 - co_2d.y))  # normalized


def setup_bg_image_nodes(rl):
    """setup all compositor nodes to render background images"""
    # https://henryegloff.com/how-to-render-a-background-image-in-blender-2-8/

    bpy.context.scene.render.film_transparent = True

    # create nodes
    tree = bpy.context.scene.node_tree
    links = tree.links
    alpha_node = tree.nodes.new(type="CompositorNodeAlphaOver")
    composite_node = tree.nodes.new(type="CompositorNodeComposite")
    scale_node = tree.nodes.new(type="CompositorNodeScale")
    image_node = tree.nodes.new(type="CompositorNodeImage")

    scale_node.space = 'RENDER_SIZE'
    scale_node.frame_method = 'CROP'

    # link nodes
    links.new(rl.outputs['Image'], alpha_node.inputs[2])
    links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    links.new(scale_node.outputs['Image'], alpha_node.inputs[1])
    links.new(alpha_node.outputs['Image'], composite_node.inputs['Image'])


def setup_camera():
    """set camera config."""
    camera = bpy.data.objects['Camera']

    # camera config
    bpy.data.cameras['Camera'].clip_start = config["camera"]["clip_start"]
    bpy.data.cameras['Camera'].clip_end = config["camera"]["clip_end"]

    # CAMERA CONFIG
    # width = cfg.resolution_x
    # height = cfg.resolution_y
    # camera.data.lens_unit = 'FOV'#'MILLIMETERS'
    if config["camera"]["lens_unit"] == 'FOV':
        camera.data.lens_unit = 'FOV'
        camera.data.angle = (config["camera"]["lens"] / 360) * 2 * math.pi
    else:
        camera.data.lens_unit = 'MILLIMETERS'
        camera.data.lens = config["camera"]["lens"]

    return camera


def get_camera_KRT(camera):
    """return 3x3 camera matrix K and 4x4 rotation, translation matrix RT.
    Experimental feature, the matrix might be wrong!"""
    # https://www.blender.org/forum/viewtopic.php?t=20231
    # Extrinsic and Intrinsic Camera Matrices
    scn = bpy.data.scenes['Scene']
    w = scn.render.resolution_x * scn.render.resolution_percentage / 100.
    h = scn.render.resolution_y * scn.render.resolution_percentage / 100.
    # Extrinsic
    RT = camera.matrix_world.inverted()
    # Intrinsic
    K = Matrix().to_3x3()
    K[0][0] = -w / 2 / math.tan(camera.data.angle / 2)
    ratio = w / h
    K[1][1] = -h / 2. / math.tan(camera.data.angle / 2) * ratio
    K[0][2] = w / 2.
    K[1][2] = h / 2.
    K[2][2] = 1.
    return K, RT


def save_camera_matrix(K):
    """save blenders camera matrix K to a file."""
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    Kdict = {
        "fx": K[0][0],
        "cx": K[0][2],
        "fy": K[1][1],
        "cy": K[1][2],
    }

    with open("/data/intermediate/render/camera_intrinsic.json", "w") as f:
        json.dump(Kdict, f)

    # save as json for better readability
    np.savetxt("/data/intermediate/render/K.txt", K)
    return Kdict


def get_sphere_coordinates(radius, inclination, azimuth):
    """convert sphere to cartesian coordinates."""
    #  https://de.m.wikipedia.org/wiki/Kugelkoordinaten
    #  radius r, inclination θ, azimuth φ)
    #  inclination [0, pi]
    #  azimuth [0, 2pi]

    x = radius * math.sin(inclination) * math.cos(azimuth)
    y = radius * math.sin(inclination) * math.sin(azimuth)
    z = radius * math.cos(inclination)
    return (x, y, z)


def place_camera(camera, radius, inclination, azimuth):
    """sample x,y,z on sphere and place camera (looking at the origin)."""

    x, y, z = get_sphere_coordinates(radius, inclination, azimuth)
    camera.location.x = x
    camera.location.y = y
    camera.location.z = z

    bpy.context.view_layer.update()

    return camera


def setup_light(temperature, key_energy, key_inc, key_azi, fill_energy, back_energy):
    """setup 3-point light model"""

    rgb = util.kelvin_to_rgb(temperature)
    d2r = math.pi / 180 #deg2rad

    key_inc *= d2r
    key_azi *= d2r

    #key light
    key_light = bpy.data.lights.new(name="key_light", type='SPOT')
    key_light.color = rgb
    key_light.energy = key_energy

    key_light_object = bpy.data.objects.new(name="key_light_object", object_data=key_light)
    bpy.context.collection.objects.link(key_light_object)
    bpy.context.view_layer.objects.active = key_light_object

    key_light_object.scale = (.1, .1, 1)
    key_light_object.location = get_sphere_coordinates(2, key_inc, key_azi)
    key_light_object.rotation_euler = ((-key_light_object.location).to_track_quat("-Z", "X").to_euler())

    #fill light
    fill_inc = math.pi/4 + key_inc / 2
    fill_azi = math.pi - key_azi

    fill_light = bpy.data.lights.new(name="fill_light", type='AREA')
    fill_light.shape = "DISK"
    fill_light.size = 1
    fill_light.color = rgb
    fill_light.energy = fill_energy

    fill_light_object = bpy.data.objects.new(name="fill_light_object", object_data=fill_light)
    bpy.context.collection.objects.link(fill_light_object)
    bpy.context.view_layer.objects.active = fill_light_object

    fill_light_object.scale = (.5, .5, 1)
    fill_light_object.location = get_sphere_coordinates(2, fill_inc, fill_azi)
    fill_light_object.rotation_euler = ((-fill_light_object.location).to_track_quat("-Z", "X").to_euler())

    #back light
    back_light = bpy.data.lights.new(name="back_light", type='POINT')
    back_light.color = rgb
    back_light.energy = back_energy
    
    back_light_object = bpy.data.objects.new(name="back_light_object", object_data=back_light)
    bpy.context.collection.objects.link(back_light_object)
    bpy.context.view_layer.objects.active = back_light_object
    
    back_light_object.location = get_sphere_coordinates(-2, 0, 0)

# def get_bg_image(bg_path=cfg.bg_paths):
#     """get list of all background images in folder 'bg_path' then choose random image."""
#     idx = random.randint(0, len(bg_path) - 1)
#
#     img_list = os.listdir(bg_path[idx])
#     randomImgNumber = random.randint(0, len(img_list) - 1)
#     bg_img = img_list[randomImgNumber]
#     bg_img_path = os.path.join(bg_path[idx], bg_img)
#     return bg_img, bg_img_path


def add_shader_on_world():
    """needed for Environment Map Background."""
    bpy.data.worlds['World'].use_nodes = True
    env_node = bpy.data.worlds['World'].node_tree.nodes.new(
        type='ShaderNodeTexEnvironment')
    emission_node = bpy.data.worlds['World'].node_tree.nodes.new(
        type='ShaderNodeEmission')
    world_node = bpy.data.worlds['World'].node_tree.nodes['World Output']

    # connect env node with emission node
    bpy.data.worlds['World'].node_tree.links.new(env_node.outputs['Color'],
                                                 emission_node.inputs['Color'])
    # connect emission node with world node
    bpy.data.worlds['World'].node_tree.links.new(
        emission_node.outputs['Emission'], world_node.inputs['Surface'])

def scene_cfg(camera, conf_obj, inc, azi):
    """configure the blender scene with specific config"""

    scene = bpy.data.scenes['Scene']

    obj = None

    files = os.listdir(os.path.join(conf_obj.model_path, conf_obj.model))

    if "model.fbx" in files:
        obj = importFBXObject(os.path.join(
            conf_obj.model_path, conf_obj.model, "model.fbx"), conf_obj)
    elif "model.obj" in files:
        obj = importOBJobject(os.path.join(
            conf_obj.model_path, conf_obj.model, "model.obj"), conf_obj)
    elif "model.ply" in files:
        obj = importPLYobject(os.path.join(
            conf_obj.model_path, conf_obj.model, "model.ply"), conf_obj)
    else:
        raise FileNotFoundError(f"{conf_obj.model}: model.(fbx|obj|ply) not in {files}")

    obj.hide_render = False

    mat = obj.active_material
    nodes = mat.node_tree.nodes

    # texture_node = nodes.get("Image Texture")
    # if texture_node:
    #    bpy.data.images.load(os.path.join(
    #        conf_obj.texture_path, texture), check_existing=True)
    #    texture_node.image = bpy.data.images[texture]

    obj.rotation_euler = (inc, azi, 0)

    # mat.node_tree.nodes["Principled BSDF"].inputs['Metallic'].default_value = metallic
    # mat.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = roughness

    camera = place_camera(
        camera,
        radius=1,
        inclination=0,
        azimuth=0)

    obj.location.x = 0
    obj.location.y = 0
    obj.location.z = 0

    # update blender object world_matrices!

    bpy.context.view_layer.update()

    # Some point in 3D you want to project
    # v = obj.location
    # Projecting v with the camera
    # K, RT = get_camera_KRT(camera)
    # p = K @ (RT @ v)
    # p /= p[2]
    # p[1] = 512 - p[1]  # openCV frame

    center = project_by_object_utils(camera, obj.location)  # object 2D center

    class_ = conf_obj.model  # class label for object
    # change order from blender to SSD paper
    corners = util.orderCorners(obj.bound_box)
    corners = np.array([np.array(project_by_object_utils(camera, obj.matrix_world @ Vector(corner))) for corner in corners])

    vertices = obj.data.vertices
    vertices = np.array([np.array(project_by_object_utils(camera, obj.matrix_world @ Vector(v.co))) for v in vertices])

    scale = np.array([config["resolution_x"], config["resolution_y"]])

    # compute bounding box either with 3D bbox or by going through vertices
    # loop through all vertices and transform to image coordinates

    annotation = dict()
    
    if conf_obj.type in ["object", "distractor"]: #always true right now

        annotation = dict(
            id=f'{conf_obj.model}-{inc}-{azi}.png',
            bbox = [0,0,0,0],
            hull = [],
        )
        
        min_x, max_x, min_y, max_y = None, None, None, None

        # higher quality, slower
        if config["compute_bbox"] == 'tight':

            # BBOX

            min_x, min_y = np.min(vertices, axis=0)
            max_x, max_y = np.max(vertices, axis=0)


            #with open("/data/intermediate/vertices.txt", "w") as f:
            #    np.savetxt(f, vertices)


            # SEGMENTATION
            faces = obj.data.polygons
            faces = [shapely.Polygon([vertices[index] for index in face.vertices]).buffer(0) for face in faces]

            hull = shapely.MultiPolygon(faces)
            hull = shapely.ops.unary_union(hull)
            hull = hull.boundary #work with boundary curves instead of full polys

            if not isinstance(hull, shapely.MultiLineString):
                hull = shapely.MultiLineString([hull])

            maxlen = max(part.length for part in hull.geoms)
            threshold = 0.01

            annotation["hull"] = [(np.array(part.coords) * np.array([config["resolution_x"], config["resolution_y"]])).tolist() for part in hull.geoms if part.length > threshold * maxlen]


        else:  # use blenders 3D bbox (simple but fast)

            labels = [class_]
            labels.append(center[0])  # center x coordinate in image space
            labels.append(center[1])  # center y coordinate in image space

            for corner in corners:
                labels.append(corner[0])
                labels.append(corner[1])

            min_x = np.min([
                labels[3], labels[5], labels[7], labels[9], labels[11],
                labels[13], labels[15], labels[17]
            ])
            max_x = np.max([
                labels[3], labels[5], labels[7], labels[9], labels[11],
                labels[13], labels[15], labels[17]
            ])

            min_y = np.min([
                labels[4], labels[6], labels[8], labels[10], labels[12],
                labels[14], labels[16], labels[18]
            ])
            max_y = np.max([
                labels[4], labels[6], labels[8], labels[10], labels[12],
                labels[14], labels[16], labels[18]
            ])


            # SEGMENTATION

            annotation["hull"] = None



        x_range = max_x - min_x
        y_range = max_y - min_y

        annotation["bbox"] = [
            min_x * config["resolution_x"], min_y * config["resolution_y"],
            x_range * config["resolution_x"], y_range * config["resolution_y"]
        ]
            



        # ROTATED BBOX

        base = vertices if config["compute_bbox"] == "tight" else corners

        up = np.array(obj.rotation_euler.to_matrix()) @ np.array([0,1,0])
        up = scale * [up[0], -up[1]] #cv coords

        up = np.array([up[0], up[1]])
        left = np.array([up[1], -up[0]])

        up /= (np.linalg.norm(up) or 1)            
        left /= (np.linalg.norm(left) or 1)


        center = (corners).mean(axis = 0)
        relative = scale * (base - center)
        projections = relative @ np.array([left, up]).T
        # = [[|vertex| * ang(left, vertex), |vertex| * ang(up, vertex)]] since left already normalized

        min_left, min_up = np.min(projections, axis=0)
        max_left, max_up = np.max(projections, axis=0)


        # log.print(f"camera rotation is {camera.rotation_euler.to_matrix()}")
        # log.print(f"number of vertices {len(base)}")
        # log.print(f"center is {center}")
        # log.print(f"rotation is {obj.rotation_euler.to_matrix()}")
        # log.print(f"up prj is {up}")
        # log.print(f"left prj is {left}")
        # log.print(f"with corners {[min_x, max_x, min_y, max_y]}")

        annotation["rotated_bbox"] = [
            (scale * center + max_left * left + max_up * up).tolist(),
            (scale * center + min_left * left + max_up * up).tolist(),
            (scale * center + min_left * left + min_up * up).tolist(),
            (scale * center + max_left * left + min_up * up).tolist(),
        ]

    return annotation


def setup():
    """one time config setup for blender."""
    bpy.ops.object.select_all(action='TOGGLE')

    # setup camera
    camera = setup_camera()

    # delete original light
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)

    # setup lights
    setup_light(**config["light"])

    # configure rendered image's parameters
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    # Bit depth per channel [8,16,32]
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.image_settings.file_format = 'PNG'  # 'PNG'
    # bpy.context.scene.render.image_settings.compression = 0  # JPEG compression
    bpy.context.scene.render.image_settings.quality = 100

    # constrain camera to look at blenders (0,0,0) scene origin (empty_object)
    camera.rotation_euler = (0,0,0)

    # composite node
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new(type="CompositorNodeRLayers")

    # setup_bg_image_nodes(rl)

    """ # save depth output file? not tested!
    if (cfg.output_depth):
        depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.base_path = ''
        depth_file_output.format.file_format = 'PNG'  # 'OPEN_EXR'
        depth_file_output.format.color_depth = '16'  # cfg.depth_color_depth
        depth_file_output.format.color_mode = 'BW'

        map_node = tree.nodes.new(type="CompositorNodeMapRange")
        map_node.inputs[1].default_value = 0  # From Min
        map_node.inputs[2].default_value = 20  # From Max
        map_node.inputs[3].default_value = 0  # To Min
        map_node.inputs[4].default_value = 1  # To Max
        links.new(rl.outputs['Depth'], map_node.inputs[0])
        links.new(map_node.outputs[0], depth_file_output.inputs[0])
    else:
        depth_file_output = None """

    # bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True

    #  delete Cube from default blender scene
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()

    #  save Model real world Bounding Box for PnP algorithm
    # np.savetxt("/intermediate/model_bounding_box.txt", util.orderCorners(obj.bound_box))

    add_shader_on_world()  # shading

    bpy.ops.wm.save_as_mainfile(filepath="/data/intermediate/render/setup.blend", check_existing=False)

    log.print("scene set up")

    return camera, None  # depth_file_output


def render_cfg():
    """setup Blenders render engine (EEVEE or CYCLES) once"""
    # refresh the list of devices
    devices = bpy.context.preferences.addons["cycles"].preferences.get_devices(
    )
    if devices:
        devices = devices[0]
        for d in devices:
            d["use"] = 1  # activate all devices
            print("activating device: " + str(d["name"]))
    if (config["use_cycles"]):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = config["samples"]
        bpy.context.scene.cycles.max_bounces = 8
        bpy.context.scene.cycles.use_denoising = config["use_cycles_denoising"]
        bpy.context.scene.cycles.use_adaptive_sampling = config["use_adaptive_sampling"]
        bpy.context.scene.cycles.adaptive_min_samples = 50
        bpy.context.scene.cycles.adaptive_threshold = 0.001
        # Intel OpenImage AI denoiser on CPU
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = config["samples"]
    if (config["use_GPU"]):
        # bpy.context.scene.render.tile_x = 64
        # bpy.context.scene.render.tile_y = 64
        bpy.context.preferences.addons[
            'cycles'].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = 'GPU'

    # https://docs.blender.org/manual/en/latest/files/media/image_formats.html
    # set image width and height
    bpy.context.scene.render.resolution_x = config["resolution_x"]
    bpy.context.scene.render.resolution_y = config["resolution_y"]


def render(camera, conf_obj: Target, reuse_existing):
    """main loop to render images"""

    render_cfg()  # setup render config once

    had_to_load = False # keep false if all reused

    annotations = []

    #  render loop
    for inc, azi in conf_obj.configs():

        filename = f'{conf_obj.model}-{inc}-{azi}.png'
        filepath = f'/data/intermediate/render/renders/{conf_obj.type}/{filename}'
        description = f'\t{round(inc*180/math.pi):>4}° {round(azi*180/math.pi):>4}°'

        if reuse_existing == True and os.path.isfile(filepath) and filename in config["old"][conf_obj.type]:
            
            annotations.append(config["old"][conf_obj.type][filename])

            log.print(f"{description} [reused]")

        else:
            had_to_load = True
            start = datetime.datetime.now()

            bpy.context.scene.render.filepath = filepath
            annotation = scene_cfg(camera, conf_obj, inc, azi)

            if conf_obj.type == "object":
                annotation["label"] = conf_obj.config["label"]

            annotations.append(annotation)

            """ if (cfg.output_depth):
                depth_file_output.file_slots[
                    0].path = bpy.context.scene.render.filepath + '_depth' """

            bpy.ops.render.render(write_still=True,
                                use_viewport=False)  # render current scene

            # for block in bpy.data.images:  # delete loaded images (bg + hdri)
            #    bpy.data.images.remove(block)

            log.print(f"{description} [in {datetime.datetime.now()-start}]")

            # save current scene as .blend file
            bpy.ops.wm.save_as_mainfile(
                filepath="/data/intermediate/render/scene.blend", check_existing=False)

    if had_to_load:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[conf_obj.model].select_set(True)
        bpy.ops.object.delete()

    return annotations


def main(reuse_existing):
    """
    call this script with 'blender --background --python main.py'

    edit the config.py file to change configuration parameters

    """
    # random.seed(cfg.seed)

    # load targets

    os.makedirs("/data/intermediate/render/", exist_ok=True)
    os.makedirs("/data/intermediate/render/renders/object", exist_ok=True)
    os.makedirs("/data/intermediate/render/renders/distractor", exist_ok=True)

    conf = {}
    with open("/data/intermediate/config/targets.json") as f:
        conf["targets"] = json.load(f)

    # log = open("/log.txt", "w")

    parser = argparse.ArgumentParser()
    parser.add_argument("--python")
    parser.add_argument("--background",
                        action="store_true")  # run blender in the background
    args = parser.parse_args()

    camera, depth_file_output = setup()  # setup once

    #render objects

    config["old"] = dict()

    def target_routine(target): #parametrized paths for object/dist

        config["old"][target] = dict()
        if os.path.isfile(f"/data/intermediate/render/renders/{target}/annotations.json"):
            with open(f"/data/intermediate/render/renders/{target}/annotations.json") as f:
                config["old"][target] = json.load(f)

        target_annotations = {}

        for target_conf in conf["targets"][target]:
            log.print(f'Rendering object {target_conf["model"]}\n')
            obj = Targets[target](target_conf)

            annotations = render(camera, obj, reuse_existing)  # render loop
            for annotation in annotations:
                target_annotations[annotation["id"]] = annotation

            del obj
            log.print(f'')

        with open(f"/data/intermediate/render/renders/{target}/annotations.json", "w") as f:
            json.dump(target_annotations, f)

    for target in ["object", "distractor"]:
        target_routine(target)

    # copy static backgrounds
    os.makedirs("/data/intermediate/backgrounds/", exist_ok=True)
    shutil.copytree("/data/input/backgrounds/static/",
                    "/data/intermediate/backgrounds/", dirs_exist_ok=True)

    # render dyn backgrounds
    #

    K, RT = get_camera_KRT(bpy.data.objects['Camera'])
    Kdict = save_camera_matrix(K)  # save Camera Matrix to K.txt


    with open("/data/intermediate/render/render.lock", "w") as f:
        f.flush()


if __name__ == '__main__':

    with open("/data/intermediate/config/log.conf", "r") as f:
        logconf = json.load(f)
        output = logconf["output"]
        reuse_existing = logconf["reuse_existing"]

    if output == "file":
        log.stdout = open("/data/log/stdout.txt", "a")
        log.stderr = open("/data/log/stderr.txt", "a")
    # won't print to terminal in any case

    try:
        main(reuse_existing)

    except Exception as e:
        log.err(traceback.format_exc())
        log.err(repr(e))
        raise e

    finally:
        if output == "file":
            log.stdout.close()
            log.stderr.close()
