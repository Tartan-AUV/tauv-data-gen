maxKeypoints = 12
frameCount = 100

WIDTH = 1600
HEIGHT = 1600

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

originRangeLow = (0.0, 0.0, 0.0)
originRangeHigh = (0.0, 0.0, 0.0)

environmentParent = None
randomStr = ""

simulation_app = None
debug = False

def init():
    global WIDTH
    global HEIGHT
    global maxKeypoints
    global frameCount
    global horizontalSymmetryPairs
    global verticalSymmetryPairs
    global classNameToID
    global classIDToName
    global classToKeypoints
    global keypointToID
    global objectsToLoad
    global enivronmentObjectsToLoad
    global textureVariants
    global simulation_app
    global originRangeLow
    global originRangeHigh
    global environmentParent
    global randomStr

    WIDTH = 1600
    HEIGHT = 1600
    maxKeypoints = 12
    frameCount = 100
    horizontalSymmetryPairs = {}
    verticalSymmetryPairs = {}
    classNameToID = {}
    classIDToName = {}    
    classToKeypoints = {}
    keypointToID = {}
    objectsToLoad = []
    enivronmentObjectsToLoad = []
    textureVariants = {}
    environmentParent = None
    randomStr = ""

    originRangeLow = (0.0, 0.0, 0.0)
    originRangeHigh = (0.0, 0.0, 0.0)