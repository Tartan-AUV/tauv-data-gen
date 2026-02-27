# import local config
import config
from config import simulation_app

# Normal python library imports
import os
from pathlib import Path
import random
from types import SimpleNamespace
from sys import argv

# standard_replicator_script.py
from isaacsim import SimulationApp

# Start the app
simulation_app_config = {
    "headless": not config.debug, # Disable GUI
    "renderer": "RaytracedLighting" if config.debug else "PathTracing",
    "width": config.WIDTH,
    "height": config.HEIGHT,
    "active_gpu": 0, # Disables all UI/Display calls
}
config.simulation_app = SimulationApp(simulation_app_config)

import omni.usd
import omni.kit.asset_converter
import omni.kit.tool.asset_importer
import omni.renderer_capture
import omni.kit.viewport
import omni.kit.commands
import omni.replicator.core as rep
import omni.syntheticdata as sd
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.semantics as sem_utils
import re

import carb
carb.settings.get_settings().set_string("/log/outputStreamLevel", "info")

# Standard USD libraries
from pxr import UsdGeom, UsdLux, Usd, Sdf, Gf, Vt, Tf, UsdSemantics, UsdShade

# Project imports
import repConversions
import setup
from outputWriter import KeypointWriter
from repUtils import *

# init warp
import warp as wp
wp.init()

def generate_data():
    # Setup scene
    rep.settings.set_stage_meters_per_unit(1.0)
    createLights()
    layer = rep.new_layer()

    if not config.debug:
        rep.settings.set_render_pathtraced(samples_per_pixel=16)

    with layer:
        camera = rep.create.camera(position=(0, 0, 0), look_at=(0, 0, 2))
        camera_paths = rep.utils.get_node_targets(camera.node, "inputs:prims")
        actual_camera_path = camera_paths[0] if camera_paths else "/Replicator/Camera_Xform/Camera"
        camera_setup(camera)

        render_product = rep.create.render_product(camera, (config.WIDTH, config.HEIGHT))

        modelRepItems = load_objects()
        environmentObjects = setup.load_environment_objects()

        # water_plane = setup_water_surface()

        all_models = rep.create.group(modelRepItems)
        environment_models = rep.create.group(environmentObjects)

        # specify behavior per frame
        if not config.debug:
            with rep.trigger.on_frame(max_execs=config.frameCount):
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
            config.simulation_app.update()

        #
        for i in range(config.frameCount):
            # Trigger the randomizers
            rep.orchestrator.step()
            
            # force a simulation update
            # This pushes the randomized poses from Replicator into the USD Stage
            config.simulation_app.update()

            # randomize textures
            for model in modelRepItems:
                prim = repConversions.replicator_item_to_prim(model)
                prim_path = repConversions.prim_to_path(prim)
                raw_class_name = get_semantic_class(prim)
                class_name = re.sub(r'_\d+$', '', raw_class_name)

                if not class_name in config.textureVariants:
                    continue

                for materialName in config.textureVariants[class_name]:
                    materialTexturePath = f"{prim_path}/Ref/_materials/{materialName}/Image_Texture"
                    texturePrim = repConversions.path_to_prim(materialTexturePath)
                    randomTexturePath = os.path.abspath(random.choice(config.textureVariants[class_name][materialName]))
                    set_unique_attribute(texturePrim, "inputs:file", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(randomTexturePath))
            
            print(f"Captured Frame {i}")

def load_objects():
    # load all the models
    modelRepItems = []
    for modelPath, _, count in config.objectsToLoad:
        full_model_path = os.path.abspath(modelPath)
        for i in range(count):
            # Create the model
            modelPrim = rep.create.from_usd(usd=full_model_path)
            modelRepItems.append(modelPrim)

    # Force Replicator to actually create these prims in the stage
    rep.orchestrator.step()
    config.simulation_app.update()

    fullIndex = 0
    for _, configPath, count in config.objectsToLoad:
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
            setup.load_training_info(actual_prim, full_config_path, i)
            fullIndex += 1
    
    return modelRepItems

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


def camera_setup(camera):
    with camera:
        # setup the camera (these are currently arbitrary, we will update once we have the actual camera calibration info)
        rep.modify.attribute("projection", "perspective")
        rep.modify.attribute("cameraProjectionType", "fisheyeOpenCV")

        rep.modify.attribute("fthetaWidth", config.WIDTH)
        rep.modify.attribute("fthetaHeight", config.HEIGHT)

        rep.modify.attribute("fthetaCx", config.WIDTH/2)
        rep.modify.attribute("fthetaCy", config.HEIGHT/2)
        
        rep.modify.attribute("openCVFx", 386.6) 
        rep.modify.attribute("openCVFy", 386.6)
        rep.modify.attribute("horizontalAperture", 20.955)
        rep.modify.attribute("verticalAperture", 20.955)
        
        rep.modify.attribute("horizontalApertureOffset", 0.0)
        rep.modify.attribute("verticalApertureOffset", 0.0)
        
        rep.modify.attribute("focalLength", 12.0)

def main():
    config.init()

    canStart = True
    if len(argv) > 1:
        canStart = setup.load_config(argv[1])
    else:
        canStart = setup.load_config()
    
    if not canStart:
        return

    generate_data()
    config.simulation_app.update() # One more update for good measure

    while config.debug:
        config.simulation_app.update()

    config.simulation_app.close() # Clean up

if __name__ == "__main__":
    main()