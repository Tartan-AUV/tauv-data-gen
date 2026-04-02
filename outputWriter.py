import warp as wp
from cv2 import fisheye
import os
import random
import re

import omni.usd
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, BackendDispatch
from omni.syntheticdata import helpers
import numpy as np

from pxr import UsdGeom, Usd

from repUtils import get_attribute

import config

class KeypointWriter(rep.Writer):
    def __init__(self, output_dir=None, camera_path=None, image_output_format="png", image_radius = 0):
        self._output_dir = output_dir
        self._image_radius = image_radius

        self.annotators = ["rgb", "bounding_box_2d_tight", "pointcloud", "camera_params", "distance_to_camera"]
        
        self._image_output_format = image_output_format
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})

        self._camera_path = camera_path

        # load keypoint prims for annotation
        self.stage = omni.usd.get_context().get_stage()
        self._frame_id = 0
    
    def initialize(self, output_dir, camera_path, image_output_format="png", image_radius = 0):
        self._output_dir = output_dir
        self._camera_path = camera_path
        self._image_output_format = image_output_format
        self._image_radius = image_radius
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.redRandAttenuate = random.uniform(0.10, 0.16)
        self.greenRandAttenuate = random.uniform(0.10, 0.14)
        self.blueRandAttenuate = random.uniform(0.09, 0.13)
        self.redRandWaterColor = random.uniform(0.05, 0.15)
        self.greenRandWaterColor = random.uniform(0.1, 0.35)
        self.blueRandWaterColor = random.uniform(0.3, 0.45)
            
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
    
    # TODO: Make not insane
    def write(self, data):
        self.camera_params = data["camera_params"]
        self.img_h, self.img_w = data["rgb"].shape[:2]
        
        bboxes = data["bounding_box_2d_tight"]["data"]
        bbox_paths = data["bounding_box_2d_tight"]["info"]["primPaths"]

        omniverse_id_to_labels = data["bounding_box_2d_tight"]["info"]["idToLabels"]

        # set up camera stuff
        self.camera_prim = self.stage.GetPrimAtPath(self._camera_path)

        # write annotation
        annotation_path = os.path.join(self._output_dir, f"rgb_{self._frame_id}.txt")
        with open(annotation_path, "w") as f:
            # Loop through detected 'main' objects
            for i, bbox in enumerate(bboxes):
                # Normalize Bbox into YOLO format: x_center, y_center, width, height
                w = (bbox['x_max'] - bbox['x_min']) / self.img_w
                h = (bbox['y_max'] - bbox['y_min']) / self.img_h
                x_center = (bbox['x_min'] / self.img_w) + (w / 2)
                y_center = (bbox['y_min'] / self.img_h) + (h / 2)

                raw_class_name = omniverse_id_to_labels[str(bbox["semanticId"])]["class"]
                class_name = re.sub(r'_\d+$', '', raw_class_name)

                # skip bounding boxes of objects we don't make annotations for
                if not class_name in config.classNameToID:
                    continue

                class_id = config.classNameToID[class_name]
                
                cropped_x = self.fullToCroppedValue(x_center)
                cropped_y = self.fullToCroppedValue(y_center)
                cropped_w = self.rescaleFullToCropped(w)
                cropped_h = self.rescaleFullToCropped(h)
                if(self.get_visible_fraction(cropped_x, cropped_y, cropped_w, cropped_h) > 0.35):
                    f.write(f"{class_id} {(cropped_x):.6f} {(cropped_y):.6f} {(cropped_w):.6f} {(cropped_h):.6f} ")
                else:
                    continue

                parent_path = bbox_paths[i]
                parent_prim = self.stage.GetPrimAtPath(parent_path)
                if not parent_prim:
                    continue
                
                # Iterate over children of bbox object (including keypoints) to find the projected coordinates of each
                projected_keypoints = {}
                for child_prim in Usd.PrimRange(parent_prim):
                    self.handlePotentialKeypoint(class_name, child_prim, projected_keypoints)

                self.fixSymmetryPairs(projected_keypoints)
                
                # Get a list based on order of ID's
                ordered_keypoints = [None] * config.maxKeypoints
                for keypoint in projected_keypoints:
                    ordered_keypoints[config.keypointToID[keypoint]] = projected_keypoints[keypoint]

                for coords in ordered_keypoints:
                    if not coords == None:
                        # write if on screen post crop
                        u, v = self.fullToCroppedPoint(coords)
                        if 0 <= u <= 1 and 0 <= v <= 1:
                            f.write(f"{u:.6f} {v:.6f} 2 ")
                        else:
                            f.write("0 0 0 ")
                    else:
                        f.write("0 0 0 ")
                
                f.write("\n")
                

        f.close()

        # write rgb image
        image_path = f"rgb_{self._frame_id}.{self._image_output_format}"
        cropped = self.cropImageToScale(data["rgb"])
        depthCropped = self.cropImageToScale(data["distance_to_camera"])
        cropped_post_processed = self.makePostProcessedRGB(cropped, depthCropped)
        self._backend.write_image(image_path, cropped_post_processed)

        # depth_path = f"depth_{self._frame_id}.{self._image_output_format}"
        # self._backend.write_image(depth_path, data["distance_to_camera"])
        self._frame_id += 1
    
    def handlePotentialKeypoint(self, class_name, child_prim, projected_keypoints):
        # skip non xforms and non keypoints
        if (not child_prim.IsA(UsdGeom.Xform)) or (not child_prim.HasAttribute("keypointName")):
            return

        child_keypoint_name = get_attribute(child_prim, "keypointName")

        # skip children without semantics
        if child_keypoint_name == None:
            return

        # skip children which are not keypoints of this class
        if not child_keypoint_name in config.classToKeypoints[class_name]:
            return

        world_transform = omni.usd.get_world_transform_matrix(child_prim)
        world_pos = world_transform.ExtractTranslation()

        projected_keypoints[child_keypoint_name] = fisheye_project(self.camera_params, 
                                                                   self.img_w, 
                                                                   self.img_h, 
                                                                   world_pos, 
                                                                   self.camera_prim)

    def fixSymmetryPairs(self, projected_keypoints):
        # fix symmetry pairs (i.e. if a "left" keypoint ends up on the right, flip them)
        for symmetricKeypoint in config.horizontalSymmetryPairs:
            (leftClass, rightClass) = config.horizontalSymmetryPairs[symmetricKeypoint]
            if leftClass in projected_keypoints and rightClass in projected_keypoints:
                uLeft, _ = projected_keypoints[leftClass]
                uRight, _ = projected_keypoints[rightClass]
                if uRight < uLeft:
                    # swap
                    projected_keypoints[leftClass], projected_keypoints[rightClass] = projected_keypoints[rightClass], projected_keypoints[leftClass]

        for symmetricKeypoint in config.verticalSymmetryPairs:
            (topClass, bottomClass) = config.verticalSymmetryPairs[symmetricKeypoint]
            if topClass in projected_keypoints and bottomClass in projected_keypoints:
                _, vBottom = projected_keypoints[topClass]
                _, vTop = projected_keypoints[bottomClass]
                # Note, higher v is lower in image
                if vBottom < vTop:
                    # swap
                    projected_keypoints[topClass], projected_keypoints[bottomClass] = projected_keypoints[topClass], projected_keypoints[bottomClass]
    
    def getCroppedSize(self):
        # the size of the image after zooming in to remove fisheye ring,
        # basically the side length of the square inscribed by that ring
        return (self._image_radius * 1.414)
    
    def fullToCroppedValue(self, value):    
        # S is scaled size
        S = self.getCroppedSize() 
        W = self.img_w # full original image size
        
        # The left edge of crop in normalized coordinates
        uv_min = (W - S) / (2 * W)
        
        # The total span of the crop in normalized units
        uv_span = S / W
        
        # Transform
        scaledValue = (value - uv_min) / uv_span
        
        return scaledValue

    def fullToCroppedPoint(self, point):
        u, v = point # Normalized 0.0 to 1.0
        return (self.fullToCroppedValue(u), self.fullToCroppedValue(v))
    
    def rescaleFullToCropped(self, value):
        # just rescales the value, but does not take the fact that the minimum has changed into account
        return value * (self.img_w / self.getCroppedSize())

    def cropImageToScale(self, image):            
        crop_h, crop_w = 858, 858
                
        start_x = self.img_w // 2 - crop_w // 2
        start_y = self.img_h // 2 - crop_h // 2
        
        cropped_img = image[start_y:start_y+crop_h, start_x:start_x+crop_w]

        return cropped_img
    
    def get_visible_fraction(self, x, y, w, h):
        # 1. Get corners of the box in the CROPPED coordinate space
        # (These values might be < 0 or > 1)
        x1, y1 = x - w/2, y - h/2
        x2, y2 = x + w/2, y + h/2

        # 2. Calculate the "Ideal" area (before clipping)
        full_area = w * h
        if full_area <= 0:
            return 0

        # 3. Clip the corners to the visible frame [0, 1]
        x1_visible = max(0, min(1, x1))
        y1_visible = max(0, min(1, y1))
        x2_visible = max(0, min(1, x2))
        y2_visible = max(0, min(1, y2))

        # 4. Calculate the visible area
        visible_width = x2_visible - x1_visible
        visible_height = y2_visible - y1_visible
        
        if visible_width <= 0 or visible_height <= 0:
            return 0
            
        visible_area = visible_width * visible_height
        
        # 5. Return the fraction of the box that is actually inside the image
        return visible_area / full_area
    
    def makePostProcessedRGB(self, colorImage, depthImage):
        # post process with depth effect

        # flatten the data before using it with warp since we don't care about x,y and this simplifies the logic
        # to access a given pixel
        rgb_flattened = colorImage.reshape(-1, 4).astype(np.float32)
        rgb_flattened /= 255.0
        depth_flattened = depthImage.reshape(-1).astype(np.float32)
        rgb_in = wp.from_numpy(rgb_flattened, dtype=wp.vec4)
        depth_in = wp.from_numpy(depth_flattened, dtype=float)
        rgb_out = wp.zeros_like(rgb_in)

        # Launch the kernel
        wp.launch(kernel=recolor_kernel, 
                  dim=len(rgb_in), 
                  inputs=[rgb_in, depth_in, rgb_out, self.redRandAttenuate, self.greenRandAttenuate, self.blueRandAttenuate, self.redRandWaterColor, self.greenRandWaterColor, self.blueRandWaterColor])

        # reshape and format into rgb image
        crop_h, crop_w = 858, 858
        processed_rgb_out = (rgb_out.numpy() * 255.0).reshape((crop_w, crop_h, 4))
        np.clip(processed_rgb_out, 0.0, 255.0)
        return processed_rgb_out.astype(np.uint8)
    
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