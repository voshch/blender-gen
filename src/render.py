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
import click

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util

log = util.Log()

config = None
with open("/data/intermediate/config/render.json") as f:
    config = json.load(f)
    config["metallic"] = []
    config["roughness"] = []


class Target:
    def __init__(self, config=None):

        if not config:
            config = {}

        if "inc" not in config:
            config["inc"] = []
        config["inc"] = config["inc"] or [0]

        if "azi" not in config:
            config["azi"] = []
        config["azi"] = config["azi"] or [0]

        if "metallic" not in config:
            config["metallic"] = []
        config["metallic"] = config["metallic"] or [0]

        if "roughness" not in config:
            config["roughness"] = []
        config["roughness"] = config["roughness"] or [1]

        if "scale" not in config:
            config["scale"] = 1

        self.model = config["model"]
        self.scale = config["scale"]

        self.config = config

    def configs(self):  # lazy iterate over all combinations
        fields = ["inc", "azi", "metallic", "roughness"]
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

# def _print(*args, **kwargs):
#     ...
# print = _print


def importPLYobject(filepath, conf_obj):
    """import PLY object from path and scale it."""

    if conf_obj.model in bpy.data.objects:

        return bpy.data.objects[conf_obj.model]

    bpy.ops.import_mesh.ply(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    obj.name = conf_obj.model
    obj.scale = (conf_obj.scale, conf_obj.scale, conf_obj.scale)  # scale PLY object

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
    config["metallic"].append(bsdf.inputs['Metallic'].default_value)
    config["roughness"].append(bsdf.inputs['Roughness'].default_value)

    return obj


def importOBJobject(filepath, conf_obj):
    """import an *.OBJ file to Blender"""

    if conf_obj.model in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[conf_obj.model].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.import_scene.obj(filepath=filepath, axis_forward='Y', axis_up='Z')
    # print("importing model with axis_forward=Y, axis_up=Z")

    # ctx = bpy.context.copy()
    # ctx['active_object'] = obj_objects[0]
    # ctx['selected_objects'] = obj_objects
    # bpy.ops.object.join(ctx)  # join multiple elements into one element
    # bpy.ops.object.join(obj_objects)  # join multiple elements into one eleme

    # get BSDF material node
    obj = bpy.context.selected_objects[0]
    obj.name = conf_obj.model

    obj.scale = (conf_obj.scale, conf_obj.scale, conf_obj.scale)

    mat = obj.active_material
    mat_links = mat.node_tree.links
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    texture = nodes.new(type="ShaderNodeTexImage")
    # mat_links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])

    # save object material inputs
    config["metallic"].append(bsdf.inputs['Metallic'].default_value)
    config["roughness"].append(bsdf.inputs['Roughness'].default_value)

    return obj

def importFBXObject(filepath, conf_obj):
    """import an *.OBJ file to Blender"""

    if conf_obj.model in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[conf_obj.model].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.import_scene.fbx(filepath=filepath, axis_forward='Y', axis_up='Z')
    # print("importing model with axis_forward=Y, axis_up=Z")

    # ctx = bpy.context.copy()
    # ctx['active_object'] = obj_objects[0]
    # ctx['selected_objects'] = obj_objects
    # bpy.ops.object.join(ctx)  # join multiple elements into one element
    # bpy.ops.object.join(obj_objects)  # join multiple elements into one eleme

    # get BSDF material node
    obj = bpy.context.selected_objects[0]
    obj.name = conf_obj.model

    obj.scale = (conf_obj.scale, conf_obj.scale, conf_obj.scale)

    mat = obj.active_material
    mat_links = mat.node_tree.links
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    texture = nodes.new(type="ShaderNodeTexImage")
    # mat_links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])

    # save object material inputs
    config["metallic"].append(bsdf.inputs['Metallic'].default_value)
    config["roughness"].append(bsdf.inputs['Roughness'].default_value)

    return obj


def leftOf(l1, l2, p):  # https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
    l1l2 = l2 - l1
    l1p = p - l1
    return l1l2[1] * l1p[0] - l1l2[0] * l1p[1] > 0


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
    # return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))
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


def setup_light(scene, inc, azi):
    """create a random point light source."""
    #  place new light in cartesian coordinates
    x, y, z = get_sphere_coordinates(
        1,
        inclination=0,
        azimuth=0)
    light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
    light_data.color = (1., 1., 1.)
    light_data.energy = config["exposure"]
    light_object = bpy.data.objects.new(
        name="my-light", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (x, y, z)


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

def euler_to_mat(roll, pitch, yaw):

    #per https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix XZY

    c1, c2, c3 = np.cos([roll, pitch, yaw])
    s1, s2, s3 = np.sin([roll, pitch, yaw])
    return np.array([
        [c2*c3,             -s2,    c2*s3           ],
        [s1*s3 + c1*c3*s2,  c1*c2,  c1*s2*s3 - c3*s1],
        [c3*s1*s2 - c1*s3,  c2*s1,  c1*c3 + s1*s2*s3]
    ])

def scene_cfg(camera, conf_obj, inc, azi, metallic, roughness):
    """configure the blender scene with specific config"""

    scene = bpy.data.scenes['Scene']
    setup_light(scene, inc, azi)

    obj = None

    files = os.listdir(os.path.join(conf_obj.model_path, conf_obj.model))

    if "model.obj" in files:
        obj = importOBJobject(os.path.join(
            conf_obj.model_path, conf_obj.model, "model.obj"), conf_obj)
    elif "model.ply" in files:
        obj = importPLYobject(os.path.join(
            conf_obj.model_path, conf_obj.model, "model.ply"), conf_obj)
    elif "model.fbx" in files:
        obj = importFBXObject(os.path.join(
            conf_obj.model_path, conf_obj.model, "model.fbx"), conf_obj)
    else:
        raise FileNotFoundError()

    obj.hide_render = False

    mat = obj.active_material
    nodes = mat.node_tree.nodes

    # texture_node = nodes.get("Image Texture")
    # if texture_node:
    #    bpy.data.images.load(os.path.join(
    #        conf_obj.texture_path, texture), check_existing=True)
    #    texture_node.image = bpy.data.images[texture]

    obj.rotation_euler = (inc, azi, 0)

    mat.node_tree.nodes["Principled BSDF"].inputs['Metallic'].default_value = metallic
    mat.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = roughness

    camera = place_camera(
        camera,
        radius=1,
        inclination=0,
        azimuth=0)

    empty_obj = bpy.data.objects["empty"]

    obj.location.x = 0
    obj.location.y = 0
    obj.location.z = 0

    rot_angle1 = 0
    rot_angle2 = 0
    rot_angle3 = 0
    # XYZ euler rotation on the empty object
    empty_obj.rotation_euler = (rot_angle1, rot_angle2, rot_angle3)

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

    # compute bounding box either with 3D bbox or by going through vertices
    # loop through all vertices and transform to image coordinates

    annotation = dict()
    
    if conf_obj.type == "object":

        annotation = dict(
            id=f'{conf_obj.model}-{inc}-{azi}-{metallic}-{roughness}.png',
            bbox = [0,0,0,0],
            hull = [],
        )

        
        
        min_x, max_x, min_y, max_y = 1, 0, 1, 0

        #bbox and segmentation
        if config["compute_bbox"] == 'tight':
            
            S = []
            highest = None

            for i, vec in enumerate(vertices):
                
                S.append(vec)

                x = vec[0]
                y = vec[1]

                if x > max_x:
                    max_x = x
                if x < min_x:
                    min_x = x
                    highest = i  # guaranteed element of convex hull
                if y > max_y:
                    max_y = y
                if y < min_y:
                    min_y = y

            hull = np.array([])  # gift wrapping algorithm for convex hull
            startpoint = S[highest]
            endpoint = None

            while True:
                hull.append(startpoint)
                endpoint = S[0]
                for point in S:
                    if (startpoint[0] == endpoint[0]) or leftOf(startpoint, endpoint, point):
                        endpoint = point
                startpoint = endpoint

                if endpoint[0] == hull[0][0]:
                    break

            annotation["hull"] = np.array(
                hull * np.array([config["resolution_x"], config["resolution_y"]])).tolist()

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

        x_range = max_x - min_x
        y_range = max_y - min_y

        annotation["bbox"] = [
            min_x * config["resolution_x"], min_y * config["resolution_y"],
            x_range * config["resolution_x"], y_range * config["resolution_y"]
        ]


        #rotated bbox

        if False:

            center = corners.mean(axis = 0)

            up = euler_to_mat(*obj.rotation_euler) @ np.array([0,1,0])
            up = np.array(project_by_object_utils(camera, obj.matrix_world @ Vector(up)))
            up /= (np.linalg.norm(up) or 1)

            left = np.array([up[1], -up[0]])
            left /= (np.linalg.norm(left) or 1)
            
            min_x, max_x = 0, 0
            min_y, max_y = 0, 0
            for vertex in vertices:
                x = np.inner(vertex, left) # = |vertex| * cos(a) since left already normalized
                y = np.inner(vertex, up)

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)


            log.print(f"number of vertices {len(vertices)}")
            log.print(f"center is {center}")
            log.print(f"rotation is {euler_to_mat(*obj.rotation_euler)}")
            log.print(f"up is {up}")
            log.print(f"left is {left}")
            log.print(f"with corners {[min_x, max_x, min_y, max_y]}")

            scale = np.array([config["resolution_x"], config["resolution_y"]])

            annotation["rotated_bbox"] = [
                (scale * (center + max_x * up + max_y * left)).tolist(),
                (scale * (center + min_x * up + max_y * left)).tolist(),
                (scale * (center + min_x * up + min_y * left)).tolist(),
                (scale * (center + max_x * up + min_y * left)).tolist(),
            ]

        else:
            annotation["rotated_bbox"] = 4*[[0,0]]


    return annotation


def setup():
    """one time config setup for blender."""
    bpy.ops.object.select_all(action='TOGGLE')
    camera = setup_camera()

    # delete Light
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)

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
    cam_constraint = camera.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_constraint.use_target_z = True
    empty_obj = bpy.data.objects.new("empty", None)
    cam_constraint.target = empty_obj

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

    #  import Model Object
    """ NumberOfObjects = 1
    for i in range(NumberOfObjects):
        if (cfg.object_paths[i][-4:] == '.obj' or
                cfg.object_paths[i][-4:] == '.OBJ'):
            obj = importOBJobject(filepath=cfg.object_paths[i])
        elif (cfg.object_paths[i][-4:] == '.ply' or
              cfg.object_paths[i][-4:] == '.PLY'):
            obj = importPLYobject(filepath=cfg.object_paths[i],
                                  scale=cfg.model_scale) """

    #  import Distractor Objects
    """ NumberOfObjects = len(cfg.distractor_paths)
    for i in range(NumberOfObjects):
        obj = importOBJobject(filepath=cfg.distractor_paths[i], distractor=True) """

    #  save Model real world Bounding Box for PnP algorithm
    # np.savetxt("/intermediate/model_bounding_box.txt", util.orderCorners(obj.bound_box))

    # add_shader_on_world()  # used for HDR background image

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


def render(camera, conf_obj, cat="unsorted"):
    """main loop to render images"""

    render_cfg()  # setup render config once

    annotations = []

    #  render loop
    for inc, azi, metallic, roughness in conf_obj.configs():

        log.print(f"\t{inc} - {azi} - {metallic} - {roughness}")

        bpy.context.scene.render.filepath = f'/data/intermediate/render/renders/{cat}/{conf_obj.model}-{inc}-{azi}-{metallic}-{roughness}.png'
        annotation = scene_cfg(camera, conf_obj,
                               inc, azi, metallic, roughness)

        if cat == "object":
            annotation["label"] = conf_obj.config["label"]

        annotations.append(annotation)

        """ if (cfg.output_depth):
            depth_file_output.file_slots[
                0].path = bpy.context.scene.render.filepath + '_depth' """

        bpy.ops.render.render(write_still=True,
                              use_viewport=False)  # render current scene

        # for block in bpy.data.images:  # delete loaded images (bg + hdri)
        #    bpy.data.images.remove(block)

        for block in bpy.data.lights:  # delete lights
            bpy.data.lights.remove(block)

    bpy.ops.wm.save_as_mainfile(
        filepath=f"/data/intermediate/render/scene-{conf_obj.model}.blend", check_existing=False)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[conf_obj.model].select_set(True)
    bpy.ops.object.delete()

    return annotations


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
def main():
    """
    call this script with 'blender --background --python main.py'

    edit the config.py file to change configuration parameters

    """
    # random.seed(cfg.seed)

    # load targets

    os.makedirs("/data/intermediate/render/", exist_ok=True)

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

    all_annotations = {}

    for obj_conf in conf["targets"]["object"]:
        log.print(f'Rendering object {obj_conf["model"]}\n')
        obj = Object(obj_conf)

        annotations = render(camera, obj, cat="object")  # render loop
        for annotation in annotations:
            all_annotations[annotation["id"]] = annotation

        del obj

    for dist_conf in conf["targets"]["distractor"]:
        log.print(f'Rendering distractor {dist_conf["model"]}\n')
        obj = Distractor(dist_conf)
        render(camera, obj, cat="distractor")  # render loop
        del obj

    # copy static backgrounds
    os.makedirs("/data/intermediate/backgrounds/", exist_ok=True)
    shutil.copytree("/data/input/backgrounds/static/",
                    "/data/intermediate/backgrounds/", dirs_exist_ok=True)

    # render dyn backgrounds
    #

    K, RT = get_camera_KRT(bpy.data.objects['Camera'])
    Kdict = save_camera_matrix(K)  # save Camera Matrix to K.txt
    # save current scene as .blend file

    with open("/data/intermediate/render/annotations.json", "w") as f:
        json.dump(all_annotations, f)

    os.makedirs("/data/intermediate/render/", exist_ok=True)
    with open("/data/intermediate/render/render.lock", "w") as f:
        f.flush()


if __name__ == '__main__':

    with open("/data/intermediate/config/log.conf", "r") as f:
        output = f.read()

    if output == "file":
        log.stdout = open("/data/log/stdout.txt", "a")
        log.stderr = open("/data/log/stderr.txt", "a")
    # won't print to terminal in any case

    try:
        main()

    except Exception as e:
        log.err(traceback.format_exc())
        log.err(repr(e))
        raise e

    finally:
        if output == "file":
            log.stdout.close()
            log.stderr.close()
