debug = False

# Normal python library imports
import os
from pathlib import Path
import glob
import json
import inspect
import random
from types import SimpleNamespace
from sys import argv

# standard_replicator_script.py
from isaacsim import SimulationApp

maxKeypoints = 12
frameCount = 100

WIDTH = 1024
HEIGHT = 1024

# Dicts to store info about keypoint symmetry,
# each maps from the name of a keypoint to a pair representing the
# the left/top item in the symmetry pair and then the right/bottom item
horizontalSymmetryPairs = {}
verticalSymmetryPairs = {}

classNameToID = {} # Maps names to ID #s
classIDToName = {}    
classToKeypoints = {} # Maps class name -> all keypoints of that class

keypointToID = {}

objectsToLoad = []
enivronmentObjectsToLoad = []

textureVariants = {} # class name -> material name -> textures which can be used with it

# Start the app
config = {
    "headless": not debug, # Disable GUI
    "renderer": "RaytracedLighting" if debug else "PathTracing",
    "width": WIDTH,
    "height": HEIGHT,
    "active_gpu": 0, # Disables all UI/Display calls
}

simulation_app = SimulationApp(config)

import omni.usd
import omni.kit.asset_converter
import omni.kit.tool.asset_importer
import omni.renderer_capture
import omni.kit.viewport
import omni.kit.commands
import omni.replicator.core as rep
import omni.syntheticdata as sd
from omni.replicator.core import AnnotatorRegistry, BackendDispatch
from omni.syntheticdata import helpers
import omni.syntheticdata as sd
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.semantics as sem_utils
import numpy as np
import re

import carb
carb.settings.get_settings().set_string("/log/outputStreamLevel", "info")

import warp as wp

from cv2 import fisheye

# Libraries for the light fix
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdGeom, UsdLux, Usd, Sdf, Gf, Vt, Tf, UsdSemantics, UsdShade # Standard USD libraries

import repConversions

# init warp
wp.init()

def generate_data():
    global objectsToLoad
    global textureVariants

    # Setup scene
    rep.settings.set_stage_meters_per_unit(1.0)
    createLights()
    layer = rep.new_layer()

    if not debug:
        rep.settings.set_render_pathtraced(samples_per_pixel=16)

    with layer:
        camera = rep.create.camera(position=(0, 0, 0), look_at=(0, 0, 2))
        camera_paths = rep.utils.get_node_targets(camera.node, "inputs:prims")
        actual_camera_path = camera_paths[0] if camera_paths else "/Replicator/Camera_Xform/Camera"
        camera_setup(camera)

        render_product = rep.create.render_product(camera, (WIDTH, HEIGHT))

        modelRepItems = load_objects()
        environmentObjects = load_environment_objects()

        # water_plane = setup_water_surface()

        all_models = rep.create.group(modelRepItems)
        environment_models = rep.create.group(environmentObjects)

        # specify behavior per frame
        if not debug:
            with rep.trigger.on_frame(max_execs=frameCount):
                with all_models:
                    # Randomize positions and rotations
                    rep.modify.pose(
                        position=rep.distribution.uniform((-3, -3, 0), (3, 3, 0)),
                        rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360))
                    )
                with camera:
                    rep.modify.pose(
                        position=rep.distribution.uniform((-8, -8, -2.5), (8, 8, 0)),
                        look_at=rep.distribution.uniform((-2, -2, -2), (2, 2, 2))
                    )

        # Initialize and attach writer
        output_dir = os.path.abspath("./output")
        writer = KeypointWriter(output_dir, actual_camera_path, "png")
        writer.initialize(
            output_dir=output_dir, 
            camera_path=actual_camera_path, 
            image_output_format="png"
        )
        writer.attach([render_product])

        # Make sure we have a couple steps to set up
        for _ in range(10):
            simulation_app.update()

        #
        for i in range(frameCount):
            # Trigger the randomizers
            rep.orchestrator.step()
            
            # force a simulation update
            # This pushes the randomized poses from Replicator into the USD Stage
            simulation_app.update()

            # randomize textures
            for model in modelRepItems:
                prim = repConversions.replicator_item_to_prim(model)
                prim_path = repConversions.prim_to_path(prim)
                raw_class_name = get_semantic_class(prim)
                class_name = re.sub(r'_\d+$', '', raw_class_name)

                if not class_name in textureVariants:
                    continue

                for materialName in textureVariants[class_name]:
                    materialTexturePath = f"{prim_path}/Ref/_materials/{materialName}/Image_Texture"
                    texturePrim = repConversions.path_to_prim(materialTexturePath)
                    randomTexturePath = os.path.abspath(random.choice(textureVariants[class_name][materialName]))
                    set_unique_attribute(texturePrim, "inputs:file", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(randomTexturePath))
            
            print(f"Captured Frame {i}")

def load_objects():
    global objectsToLoad
    # load all the models
    modelRepItems = []
    for modelPath, _, count in objectsToLoad:
        full_model_path = os.path.abspath(modelPath)
        for i in range(count):
            # Create the model
            modelPrim = rep.create.from_usd(usd=full_model_path)
            modelRepItems.append(modelPrim)

    # Force Replicator to actually create these prims in the stage
    rep.orchestrator.step()
    simulation_app.update()

    fullIndex = 0
    for _, configPath, count in objectsToLoad:
        full_config_path = os.path.abspath(configPath)
        
        for i in range(count):
            # Get the Replicator Item
            rep_item = modelRepItems[fullIndex]
            
            prim_path = repConversions.replicator_item_to_path(rep_item)
            
            if not prim_path:
                print(f"[ERROR] Could not resolve path for model {fullIndex}")
                fullIndex += 1
                continue
            
            # Get the actual USD Prim object
            actual_prim = repConversions.path_to_prim(prim_path)
            load_training_info(actual_prim, full_config_path, i)
            fullIndex += 1
    
    return modelRepItems

def load_training_info(prim, jsonPath, instanceNumber):
    jsonParse = json.loads(Path(jsonPath).read_text())

    if not "classID" in jsonParse:
        print(f"Missing classID in file {jsonPath}!")
        return
    if not "className" in jsonParse:
        print(f"Missing className in file {jsonPath}!")
        return
    
    classID = jsonParse["classID"]
    className = jsonParse["className"]

    if classID in classIDToName:
        if classIDToName[classID] != className:
            print(f"Class ID {classID} conflict!")
            return
    
    # update class related dicts
    classNameToID[className] = classID
    classIDToName[classID] = className
    classToKeypoints[className] = set()
    
    with repConversions.prim_to_replicator_item(prim):
        rep.modify.semantics([('class', f"{className}_{instanceNumber}")], mode='clear')

    if "keypoints" in jsonParse:
        for keypointName in jsonParse["keypoints"]:
            keypoint = jsonParse["keypoints"][keypointName]

            if(not "position" in keypoint):
                print(f"Keypoint {keypointName} in {className} is missing a position!")
                continue
            if(not "keypointID" in keypoint):
                print(f"Keypoint {keypointName} in {className} is missing an ID!")
                continue
            if(keypointName in keypointToID and keypointToID[keypointName] != keypoint["keypointID"]):
                print(f"Duplicated keypoint name {keypointName}!")
                continue

            add_keypoint(prim, 
                         keypointName, 
                         tuple(keypoint["position"]),
                         instanceNumber)
            
            # Update ID dict
            keypointToID[keypointName] = keypoint["keypointID"]
            
            # Update symmetry
            if "symmetryMode" in keypoint:
                if(keypoint["symmetryMode"] == "horizontalSymmetry"):
                    update_horizontal_symmetry(keypoint, keypointName)
                if(keypoint["symmetryMode"] == "verticalSymmetry"):
                    update_vertical_symmetry(keypoint, keypointName)

            # update keypoint/classes map
            classToKeypoints[className].add(keypointName)
        
        if "textureVariants" in jsonParse:
            textureVariants[className] = {}
            for variant in jsonParse["textureVariants"]:
                if not "materialName" in variant:
                    print(f"Missing material name for texture variant in {className}!")
                
                materialName = variant["materialName"]
                availableTextures = variant["availableTextures"]
                textureVariants[className][materialName] = availableTextures

def add_keypoint(targetPrim, name, position, instanceNumber):
    keypointRepItem = None

    with rep.get.prims(path_pattern=str(targetPrim.GetPath())):
        keypointRepItem = rep.create.xform(
            position = position, 
            parent = targetPrim
        )

    with keypointRepItem:
        rep.modify.semantics([('class', f"{name}_{instanceNumber}")], mode='clear')
        rep.modify.attribute("purpose", "guide")    
        
    set_unique_attribute(repConversions.replicator_item_to_prim(keypointRepItem), "keypointName", Sdf.ValueTypeNames.String, f"{name}")        

        
def update_horizontal_symmetry(keypoint, keypointName):
    update_symmetry(keypoint, keypointName, horizontalSymmetryPairs, "left", "right")

def update_vertical_symmetry(keypoint, keypointName):
    update_symmetry(keypoint, keypointName, verticalSymmetryPairs, "top", "bottom")

def update_symmetry(keypoint, keypointName, symmetricDict, category1, category2):
    # validate input
    if(not "symmetryDetails" in keypoint):
        print(f"{keypointName} has a symmetryMode but no symmetryDetails!")
        return
    
    symmetryDetails = keypoint["symmetryDetails"]

    if(not "side" in symmetryDetails):
        print(f"{keypointName} has no side in its symmetryDetails!")
        return
    if(not "symmetricWith" in symmetryDetails):
        print(f"{keypointName} has no symmetricWith in its symmetryDetails!")
        return
    if(not (symmetryDetails["side"] == category1 or symmetryDetails["side"] == category2)):
        print(f"{keypointName} has an invalid side in its symmetryDetails!")
        return
    
    # update dict
    symmetricWith = symmetryDetails["symmetricWith"]

    if(keypoint["symmetryDetails"]["side"] == category1):
        symmetricDict[keypointName] = (keypointName, symmetricWith)
        symmetricDict[symmetricWith] = (keypointName, symmetricWith)
    if(keypoint["symmetryDetails"]["side"] == category2):
        symmetricDict[keypointName] = (symmetricWith, keypointName)
        symmetricDict[symmetricWith] = (symmetricWith, keypointName)

def load_environment_objects():
    global enivronmentObjectsToLoad

    environmentObjects = []

    print(enivronmentObjectsToLoad)
    for path, position, euler, scale in enivronmentObjectsToLoad:
        print(f"path {path} | position {position} | euler {euler} | scale {scale}")
        repItem = rep.create.from_usd(
            usd=path,
            position=position,
            rotation=euler,
            scale=scale,
            name="test"
        )
        with repItem:
            rep.modify.semantics([('class', f"environmentObject")], mode='clear')

        environmentObjects.append(repItem)

    rep.orchestrator.step()
    simulation_app.update()

    return environmentObjects

def setup_water_surface():
    water_plane = rep.create.plane(scale=60, position=(0, 0, 0))

    water_material = rep.create.material_omnipbr()

    shaderPath = repConversions.replicator_item_to_path(water_material)+"/Shader"
    shaderPrim = repConversions.path_to_prim(shaderPath)
    set_unique_attribute(shaderPrim, "info:mdl:sourceAsset", Sdf.ValueTypeNames.String, "OmniGlass.mdl")
    set_unique_attribute(shaderPrim, "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.String, "OmniGlass")
    set_unique_attribute(shaderPrim, "inputs:glass_ior", Sdf.ValueTypeNames.Float, 1.33)
    set_unique_attribute(shaderPrim, "inputs:thin_walled", Sdf.ValueTypeNames.Bool, True)
    set_unique_attribute(shaderPrim, "inputs:reflection_color", Sdf.ValueTypeNames.Color3f, (1.0, 1.0, 1.0))

    with water_plane:
        rep.modify.material(water_material)
    
    return water_plane

def validate_usd_asset(path):
    """Checks if the USD file exists and has a valid Default Prim."""
    # 1. Physical Check
    if not os.path.exists(path):
        print(f"[ERROR] File does not exist at: {path}")
        return False
    
    # 2. Structure Check
    stage = Usd.Stage.Open(path)
    if not stage:
        print(f"[ERROR] Could not open stage at {path}. File might be corrupted.")
        return False
    
    if not stage.HasDefaultPrim():
        print(f"[WARNING] {path} has no Default Prim set!")
        print(">> Logic: Replicator needs a Default Prim to reference the asset.")
        # List available prims to help you debug
        prims = [p.GetName() for p in stage.GetPseudoRoot().GetChildren()]
        print(f">> Available root prims: {prims}")
        return False
        
    print(f"[SUCCESS] Asset {path} is valid. Default Prim: {stage.GetDefaultPrim().GetName()}")
    return True

def fix_default_prim(path):
    """Ensures the USD at 'path' has a Default Prim set."""
    stage = Usd.Stage.Open(path)
    if not stage:
        return False
    
    if not stage.HasDefaultPrim():
        # Get the first child
        root_prims = stage.GetPseudoRoot().GetChildren()
        if root_prims:
            first_prim = root_prims[0]
            stage.SetDefaultPrim(first_prim)
            stage.GetRootLayer().Save()
            print(f"[FIXED] Set '{first_prim.GetName()}' as Default Prim in {path}")
            return True
        else:
            print(f"[ERROR] {path} is empty. No prims found to set as default.")
            return False
    return True

def createLights():
    # We create lights like this because IsaacSim v5 has an issue where
    # creating lights within a with rep.new_layer() statement caps the
    # light intensity at 1.0 (very dark)
    stage = omni.usd.get_context().get_stage()
    distant_path = "/World/DistantLight"
    distant_light = UsdLux.DistantLight.Define(stage, distant_path)
    distant_light.CreateIntensityAttr(700.0)
    distant_light.CreateColorAttr((1.0, 1.0, 0.95))
    prim = stage.GetPrimAtPath(distant_path)
    UsdGeom.XformCommonAPI(prim).SetRotate((315.0, 0.0, 0.0))
    UsdGeom.XformCommonAPI(prim).SetTranslate((0.0, 8.0, 0.0))

    ambient_path = "/World/AmbientLight"
    dome_light = UsdLux.DomeLight.Define(stage, ambient_path)
    dome_light.CreateIntensityAttr(100.0)
    dome_light.CreateColorAttr((1.0, 1.0, 0.95))

    # skyPath = os.path.abspath("./citrus_orchard_road_puresky_4k.exr")
    # dome_light.CreateTextureFileAttr(Sdf.AssetPath(skyPath))
    # dome_light.CreateTextureFormatAttr("latlong")


def camera_setup(camera):
    with camera:
        # setup the camera (these are currently arbitrary, we will update once we have the actual camera calibration info)
        rep.modify.attribute("projection", "perspective")
        rep.modify.attribute("cameraProjectionType", "fisheyeOpenCV")

        rep.modify.attribute("fthetaWidth", WIDTH)
        rep.modify.attribute("fthetaHeight", HEIGHT)

        rep.modify.attribute("fthetaCx", WIDTH/2)
        rep.modify.attribute("fthetaCy", HEIGHT/2)
        
        rep.modify.attribute("openCVFx", 386.6) 
        rep.modify.attribute("openCVFy", 386.6)
        rep.modify.attribute("horizontalAperture", 20.955)
        rep.modify.attribute("verticalAperture", 20.955)
        
        rep.modify.attribute("horizontalApertureOffset", 0.0)
        rep.modify.attribute("verticalApertureOffset", 0.0)
        
        rep.modify.attribute("focalLength", 12.0)

class KeypointWriter(rep.Writer):
    def __init__(self, output_dir=None, camera_path=None, image_output_format="png"):
        self._output_dir = output_dir

        self.annotators = ["rgb", "bounding_box_2d_tight", "pointcloud", "camera_params", "distance_to_camera"]
        
        self._image_output_format = image_output_format
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})

        self._camera_path = camera_path

        # load keypoint prims for annotation
        self.stage = omni.usd.get_context().get_stage()
        self._frame_id = 0
    
    def initialize(self, output_dir, camera_path, image_output_format="png"):
        self._output_dir = output_dir
        self._camera_path = camera_path
        self._image_output_format = image_output_format
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
    
    # TODO: Make not insane
    def write(self, data):
        global classNameToID

        # post process with depth effect

        # flatten the data before using it with warp since we don't care about x,y and this simplifies the logic
        # to access a given pixel
        rgb_flattened = data["rgb"].reshape(-1, 4).astype(np.float32)
        rgb_flattened /= 255.0
        depth_flattened = data["distance_to_camera"].reshape(-1).astype(np.float32)
        rgb_in = wp.from_numpy(rgb_flattened, dtype=wp.vec4)
        depth_in = wp.from_numpy(depth_flattened, dtype=float)
        rgb_out = wp.zeros_like(rgb_in)

        # scramble things a little
        redRandAttenuate = random.uniform(0.10, 0.16)
        greenRandAttenuate = random.uniform(0.10, 0.14)
        blueRandAttenuate = random.uniform(0.09, 0.13)
        redRandWaterColor = random.uniform(0.05, 0.15)
        greenRandWaterColor = random.uniform(0.1, 0.35)
        blueRandWaterColor = random.uniform(0.3, 0.45)

        # Launch the kernel
        wp.launch(kernel=recolor_kernel, 
                  dim=len(rgb_in), 
                  inputs=[rgb_in, depth_in, rgb_out, redRandAttenuate, greenRandAttenuate, blueRandAttenuate, redRandWaterColor, greenRandWaterColor, blueRandWaterColor])
        
        camera_params = data["camera_params"]
        img_h, img_w = data["rgb"].shape[:2]
        
        bboxes = data["bounding_box_2d_tight"]["data"]
        bbox_paths = data["bounding_box_2d_tight"]["info"]["primPaths"]

        omniverse_id_to_labels = data["bounding_box_2d_tight"]["info"]["idToLabels"]

        # set up camera stuff
        camera_prim = self.stage.GetPrimAtPath(self._camera_path)

        # write annotation
        annotation_path = os.path.join(self._output_dir, f"label_{self._frame_id}.txt")
        with open(annotation_path, "w") as f:
            # Loop through detected 'main' objects
            for i, bbox in enumerate(bboxes):
                # Normalize Bbox into YOLO format: x_center, y_center, width, height
                w = (bbox['x_max'] - bbox['x_min']) / img_w
                h = (bbox['y_max'] - bbox['y_min']) / img_h
                x_center = (bbox['x_min'] / img_w) + (w / 2)
                y_center = (bbox['y_min'] / img_h) + (h / 2)

                raw_class_name = omniverse_id_to_labels[str(bbox["semanticId"])]["class"]
                class_name = re.sub(r'_\d+$', '', raw_class_name)

                # skip bounding boxes of objects we don't make annotations for
                if not class_name in classNameToID:
                    continue

                class_id = classNameToID[class_name]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} ")

                parent_path = bbox_paths[i]
                parent_prim = self.stage.GetPrimAtPath(parent_path)
                if not parent_prim:
                    continue
                
                # Iterate over children of bbox object (including keypoints) to find the projected coordinates of each
                projected_keypoints = {}
                for child_prim in Usd.PrimRange(parent_prim):
                    # skip non xforms and non keypoints
                    if (not child_prim.IsA(UsdGeom.Xform)) or (not child_prim.HasAttribute("keypointName")):
                        continue

                    child_keypoint_name = get_attribute(child_prim, "keypointName")

                    # skip children without semantics
                    if child_keypoint_name == None:
                        continue

                    # skip children which are not keypoints of this class
                    if not child_keypoint_name in classToKeypoints[class_name]:
                        continue

                    world_transform = omni.usd.get_world_transform_matrix(child_prim)
                    world_pos = world_transform.ExtractTranslation()

                    projected_keypoints[child_keypoint_name] = fisheye_project(camera_params, img_w, img_h, world_pos, camera_prim)

                # fix symmetry pairs (i.e. if a "left" keypoint ends up on the right, flip them)
                for symmetricKeypoint in horizontalSymmetryPairs:
                    (leftClass, rightClass) = horizontalSymmetryPairs[symmetricKeypoint]
                    if leftClass in projected_keypoints and rightClass in projected_keypoints:
                        uLeft, _ = projected_keypoints[leftClass]
                        uRight, _ = projected_keypoints[rightClass]
                        if uRight < uLeft:
                            # swap
                            projected_keypoints[leftClass], projected_keypoints[rightClass] = projected_keypoints[rightClass], projected_keypoints[leftClass]

                for symmetricKeypoint in verticalSymmetryPairs:
                    (topClass, bottomClass) = verticalSymmetryPairs[symmetricKeypoint]
                    if topClass in projected_keypoints and bottomClass in projected_keypoints:
                        _, vBottom = projected_keypoints[topClass]
                        _, vTop = projected_keypoints[bottomClass]
                        # Note, higher v is lower in image
                        if vBottom < vTop:
                            # swap
                            projected_keypoints[topClass], projected_keypoints[bottomClass] = projected_keypoints[topClass], projected_keypoints[bottomClass]
                
                # Get a list based on order of ID's
                ordered_keypoints = [None] * maxKeypoints
                for keypoint in projected_keypoints:
                    ordered_keypoints[keypointToID[keypoint]] = projected_keypoints[keypoint]

                for coords in ordered_keypoints:
                    if not coords == None:
                        u, v = coords
                        # write if on screen
                        if 0 <= u <= 1 and 0 <= v <= 1:
                            f.write(f"{u:.6f} {v:.6f} 2 ")
                        else:
                            f.write("0 0 0 ")
                    else:
                        f.write("0 0 0 ")
                
                f.write("\n")
                

        f.close()

        # write rgb image
        processed_rgb_out = (rgb_out.numpy() * 255.0).reshape((HEIGHT, WIDTH, 4))
        np.clip(processed_rgb_out, 0.0, 255.0)
        processed_rgb_out = processed_rgb_out.astype(np.uint8)
        image_path = f"rgb_{self._frame_id}.{self._image_output_format}"
        self._backend.write_image(image_path, processed_rgb_out)
        # depth_path = f"depth_{self._frame_id}.{self._image_output_format}"
        # self._backend.write_image(depth_path, data["distance_to_camera"])
        self._frame_id += 1

# a warp kernel to recolor images to apply the underwater effect
@wp.kernel
def recolor_kernel(rgb: wp.array(dtype=wp.vec4), 
                   depth: wp.array(dtype=float), 
                   out_rgb: wp.array(dtype=wp.vec4),
                   attenuationR: float,
                   attenuationG: float,
                   attenuationB: float,
                   finalR: float,
                   finalG: float,
                   finalB: float):
    # Underwater image recoloring based on https://arxiv.org/html/2503.01074v2

    tid = wp.tid()
    
    depthToPixel = depth[tid]
    rWeight = wp.exp(-attenuationR * depthToPixel)
    gWeight = wp.exp(-attenuationG * depthToPixel)
    bWeight = wp.exp(-attenuationB * depthToPixel)

    out_rgb[tid][0] = rgb[tid][0]*rWeight + finalR*(1.0-rWeight)
    out_rgb[tid][1] = rgb[tid][1]*gWeight + finalG*(1.0-gWeight)
    out_rgb[tid][2] = rgb[tid][2]*bWeight + finalB*(1.0-bWeight)
 
    out_rgb[tid][3] = 1.0

# Projects using OpenCV fisheye parameters in camera_params
def fisheye_project(camera_params, img_w, img_h, world_pos, camera_prim):
    # Get the LIVE World-to-Camera matrix (The fix for randomization)
    camera_world_transform = omni.usd.get_world_transform_matrix(camera_prim)
    view_mat = np.array(camera_world_transform.GetInverse()).reshape(4, 4)
    
    # Extract Intrinsics (K) and Distortion (D) from params
    fx = camera_params["cameraOpenCVFx"]
    fy = camera_params["cameraOpenCVFy"]
    cx, cy = camera_params["cameraFisheyeOpticalCentre"]
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    D = camera_params["cameraFisheyePolynomial"][:4].astype(np.float32)

    # Transform Point to Camera Local Space (3D)
    world_pos_4d = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
    p_view_4d = world_pos_4d @ view_mat
    p_cam = p_view_4d[:3] / p_view_4d[3]

    # Coordinate System Swap
    # USD: Right(+X), Up(+Y), Back(+Z)
    # OpenCV: Right(+X), Down(+Y), Forward(+Z)
    p_cv = np.array([p_cam[0], -p_cam[1], -p_cam[2]], dtype=np.float32)

    # Fisheye Projection
    if p_cv[2] <= 0: # Point is behind the lens
        return (-1.0, -1.0)

    # We already transformed the point, so rvec and tvec are zero
    rvec = tvec = np.zeros(3, dtype=np.float32)
    
    # projectPoints expects a (N, 1, 3) array
    image_points, _ = fisheye.projectPoints(
        p_cv.reshape(1, 1, 3), rvec, tvec, K, D
    )
    
    px, py = image_points[0][0]
    
    # Return normalized (0-1) for YOLO
    return (px / img_w, py / img_h)

def get_semantic_class(prim):
    attr = prim.GetAttribute("semantics:labels:class")
    if attr.HasValue():
        val = attr.Get()
        # Check if it's an array (list-like) or a single token
        if hasattr(val, "__len__") and len(val) > 0:
        # Always get the last token
            return str(val[-1])
        elif val:
            return str(val)
    return None

def set_unique_attribute(prim, attribute, type, value):
    if not prim:
        return

    if not prim.HasAttribute(attribute):
        prim.CreateAttribute(attribute, type)

    attr = prim.GetAttribute(attribute)
    attr.Clear()
    attr.Set(value)

def get_attribute(prim, attribute):
    return prim.GetAttribute(attribute).Get()

def load_config(configFile="./config.json"):
    global maxKeypoints
    global objectsToLoad
    global enivronmentObjectsToLoad
    global frameCount

    jsonParse = json.loads(Path(os.path.abspath(configFile)).read_text())

    if not "maxKeypoints" in jsonParse:
        print(f"Config file {configFile} is missing required field 'maxKeypoints'!")
        return False
    
    maxKeypoints = jsonParse["maxKeypoints"]

    if "frames" in jsonParse:
        frameCount = jsonParse["frames"]

    if "objects" in jsonParse:
        objectsList = jsonParse["objects"]
        for obj in objectsList:
            if not ("model" in obj):
                print(f"Config file {configFile} contains an object which lacks a model path!")
                return False
            
            if "count" in obj:
                objectsToLoad.append((obj["model"], obj["config"], obj["count"]))
            else:
                objectsToLoad.append((obj["model"], obj["config"], 1))
    
    if "environmentObjects" in jsonParse:
        enivornmentObjectsList = jsonParse["environmentObjects"]
        for envObj in enivornmentObjectsList:
            if not ("model" in envObj):
                print(f"Config file {configFile} contains an environment object which lacks a model path!")
                return False
            
            path = envObj["model"]
            
            position = (0.0, 0.0, 0.0)
            euler = (0.0, 0.0, 0.0)
            scale = (1.0, 1.0, 1.0)

            if "position" in envObj:
                position = tuple(envObj["position"])
            if "euler" in envObj:
                euler = tuple(envObj["euler"])
            if "scale" in envObj:
                scale = tuple(envObj["scale"])
            
            enivronmentObjectsToLoad.append((os.path.abspath(path), position, euler, scale))

    return True

def main():
    canStart = True
    if len(argv) > 1:
        canStart = load_config(argv[1])
    else:
        canStart = load_config()
    
    if not canStart:
        return

    generate_data()
    simulation_app.update() # One more update for good measure

    while debug:
        simulation_app.update()

    simulation_app.close() # Clean up

if __name__ == "__main__":
    main()