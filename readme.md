This is the TAUV submodule responsible for generating labelled synthetic data for computer vision model training.

## Setup:
- Requires python3.11 (exactly this version)
- Requires Omniverse and IsaacSim Packages: `pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com` (I recommend you create a venv in the outermost directory of the project)
- Requires usd-core: `pip install usd-core`
- Requires opencv-python: `pip install opencv-python`
- Requires numpy `pip install numpy`
- Requires warp `pip install warp-lang`
- Recommended that you run `python3.11 -m isaacsim --generate-vscode-settings` so vscode can provide autocomplete for omni libraries (although this is often broken)

## Usage:
`python3.11 main.py` Output images and annotations are in the ./output directory <br>
`python3.11 checker.py <image index>` will display the index-th image in output with its annotations overlaid.

## Main Config File Options:
- **frames (int, required):** Number of frames to render
- **maxKeypoints (int, required):** Maximum number of keypoints any object can have
- **objects (list of objects):** The list of objects which we want to randomize and generate annotations for. The data for each object in this list consists of:
    - **model (string, required):** A path to the object's usdz model file
    - **config (string, required):** A path to the object's usdz config file
    - **count (int)**: The number of this object to create. Defaults to 1.
- **environmentObjects (list of environmentObjects):** The list of "environment objects" (models which make up the environment, do not move, and do not generate annotations). The data for each environmentObject in this list consists of:
    - **model (string, required):** A path to the environmentObject's usdz model file.
    - **position (list of 3 reals):** The position this environmentObject should be placed at. Defaults to (0,0,0)
    - **euler (list of 3 reals):** The euler angle rotation this environmentObject will have. Defaults to (0,0,0)
    - **scale (list of 3 reals):** The scale that will be applied to this environment object. Defaults to (1,1,1)

## Object Config File Options:
- **classID (int, required):** The unique number which corresponds to this class which will be used in annotation generation
- **className (string, required):** The name of this object's class
- **keypoints (list of keypoints):** The list of keypoints of this class. The keypoint format is explained further down
- **textureVariants (list of textureVariants):** The list of texture variant objects which allow us to randomize the textures on the object each frame. The data for each textureVariant in this list consists of:
    - **materialName (string, required):** The name of the usd material which needs to be updated to modify the appearance of this object or the part of it which contains the texture. This usually be set in advance when exporting from 3D modelling software, but it can also be found in the omniverse editor
    - **availableTextures (list of strings, required):** Each string in this list is a path to an image file which will be used to randomize the appearance of this material

## Keypoint Options:
- **position (list of 3 reals, required):** The position of this keypoint relative to the object's origin.
- **keypointID (int, required):** A unique integer (at least within this class) which will be used in annotation generation
- **symmetryMode (string):** Specifies what kind of symmetry this keypoint has with another keypoint: horizontalSymmetry, verticalSymmetry, or none. By specifying a symmetry mode, we can ensure that the model always learns to put the correct keypoint on the correct side of the object, regardless of its orientation.
- **symmetryDetails (required if using symmetryMode):**
    - **side (string, required):** Specifies the side this keypoint is on in a symmetry pair. left and right for horizontal pairs. top and bottom for vertical pairs
    - **symmetricWith (string, required):** The name of the other keypoint this keypoint is symmetric with

## Important Concepts
- **Prim vs RepItem vs. Path:** Prims are what Omniverse (and the Pixar USD file format which it uses) call its fundamental objects. Generally for our purposes these are 3D models, but lights, cameras, and materials are also prims. RepItems are wrappers around prims that Replicator (the library under Omniverse used in synthetic data generation) uses for programmer convenience and effiency under the hood. Most tasks related to randomization can be accomplished using RepItems (and should be), but sometimes we need the underlying Prim instead. Conversions functions between RepItems, Prims, and Paths (to prims), can be found in repConversions.py
- **Exporting Models:** Models should be exported in usdz format as these have by far the best omniverse support. Unfortunately, CAD software does not typically support this as an export option, so the model must be re-exported from Blender. While in Blender, check that the models UV mapping is correct and that its materials have useful names. These issues are easier to fix here and will streamline using Omniverse
- **Debug Mode:** Enabling debug mode enables the Omniverse GUI and decreases the lighting quality of the rendered images. In addition to debugging, debug mode is useful for finding the names of Prim attributes in Omniverse and setting the locations of keypoints.

## TODO:
**Features:**
- Improve underwater appearance by getting a proper water material instead of just attenuating towards a constant color.
- Add a check to make sure keypoints aren't obstructed before writing them. This would involve comparing the depth of the keypoint to the depth listed at the same projected point in the depth map and seeing if it's within an acceptable tolerance which we would add to the keypoint config.
- Add distractor objects.
- Add a wider variety of objects to detect.

**Structure:**
- Reduce use of global variables (ideally to 0)
- Break down functions which are too long or deeply nested (most notably KeypointWriter.write)
- Split main.py into separate files for object loading, randomization, annotation writing, etc.