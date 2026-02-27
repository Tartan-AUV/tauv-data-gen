import config
from config import simulation_app

import glob
import json
import inspect
import os
from pathlib import Path

import omni.usd
import omni.replicator.core as rep
from isaacsim import SimulationApp


import repConversions
from repUtils import *

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

    if classID in config.classIDToName:
        if config.classIDToName[classID] != className:
            print(f"Class ID {classID} conflict!")
            return
    
    # update class related dicts
    config.classNameToID[className] = classID
    config.classIDToName[classID] = className
    config.classToKeypoints[className] = set()
    
    with repConversions.prim_to_replicator_item(prim):
        rep.modify.semantics([('class', f"{className}_{instanceNumber}")], mode='clear')

    if "keypoints" in jsonParse:
        for keypointName in jsonParse["keypoints"]:
            keypoint = jsonParse["keypoints"][keypointName]
            make_keypoint(prim, className, keypoint, keypointName, instanceNumber)
    
    if "textureVariants" in jsonParse:
        config.textureVariants[className] = {}
        for variant in jsonParse["textureVariants"]:
            if not "materialName" in variant:
                print(f"Missing material name for texture variant in {className}!")
            
            materialName = variant["materialName"]
            availableTextures = variant["availableTextures"]
            config.textureVariants[className][materialName] = availableTextures

            

def make_keypoint(classPrim, className, keypoint, keypointName, instanceNumber):
    if(not "position" in keypoint):
        print(f"Keypoint {keypointName} in {className} is missing a position!")
        return
    if(not "keypointID" in keypoint):
        print(f"Keypoint {keypointName} in {className} is missing an ID!")
        return
    if(keypointName in config.keypointToID and config.keypointToID[keypointName] != keypoint["keypointID"]):
        print(f"Duplicated keypoint name {keypointName}!")
        return

    add_keypoint_to_target(classPrim, 
                            keypointName, 
                            tuple(keypoint["position"]),
                            instanceNumber)
    
    # Update ID dict
    config.keypointToID[keypointName] = keypoint["keypointID"]
    
    # Update symmetry
    if "symmetryMode" in keypoint:
        if(keypoint["symmetryMode"] == "horizontalSymmetry"):
            update_horizontal_symmetry(keypoint, keypointName)
        if(keypoint["symmetryMode"] == "verticalSymmetry"):
            update_vertical_symmetry(keypoint, keypointName)

    # update keypoint/classes map
    config.classToKeypoints[className].add(keypointName)

def add_keypoint_to_target(targetPrim, name, position, instanceNumber):
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
    update_symmetry(keypoint, keypointName, config.horizontalSymmetryPairs, "left", "right")

def update_vertical_symmetry(keypoint, keypointName):
    update_symmetry(keypoint, keypointName, config.verticalSymmetryPairs, "top", "bottom")

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
    environmentObjects = []

    print(config.enivronmentObjectsToLoad)
    for path, position, euler, scale in config.enivronmentObjectsToLoad:
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

def load_config(configFile="./config.json"):
    jsonParse = json.loads(Path(os.path.abspath(configFile)).read_text())

    if not "maxKeypoints" in jsonParse:
        print(f"Config file {configFile} is missing required field 'maxKeypoints'!")
        return False
    
    config.maxKeypoints = jsonParse["maxKeypoints"]

    if "frames" in jsonParse:
        config.frameCount = jsonParse["frames"]

    if "objects" in jsonParse:
        objectsList = jsonParse["objects"]
        for obj in objectsList:
            if not ("model" in obj):
                print(f"Config file {configFile} contains an object which lacks a model path!")
                return False
            
            if "count" in obj:
                config.objectsToLoad.append((obj["model"], obj["config"], obj["count"]))
            else:
                config.objectsToLoad.append((obj["model"], obj["config"], 1))
    
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
            
            config.enivronmentObjectsToLoad.append((os.path.abspath(path), position, euler, scale))

    return True