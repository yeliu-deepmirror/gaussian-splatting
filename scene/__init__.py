#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import gc
import random
from random import randint
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, loadCam

# the version read image from file one by one, to save memory consumption
# refer to the original repo for the raw version
class Scene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=-1, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args
        self.resolution_scales = resolution_scales
        self.iteration_offset = 0

        if load_iteration is not None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            if self.loaded_iter is not None:
                self.iteration_offset = self.loaded_iter
                print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.front_only, args.cut_ratio)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.nerf_normalization = scene_info.nerf_normalization
        self.train_cameras = scene_info.train_cameras
        self.test_cameras = scene_info.test_cameras
        print("  - Nerf Normalization:", scene_info.nerf_normalization)
        print("  - Load", len(self.train_cameras), "Training Cameras")
        print("  - Load", len(self.test_cameras), "Test Cameras")
        self.reset_train_test_ids()

        if not self.args.load_dynamic:
            print("  - Load all the images.")
            self.train_cameras_list = {}
            self.test_cameras_list = {}
            for resolution_scale in resolution_scales:
                self.train_cameras_list[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, "train")
                self.test_cameras_list[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, "test")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        del scene_info
        gc.collect()

    def max_height(self):
        return self.nerf_normalization["cameras_max"][2]

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration + self.iteration_offset))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def reset_train_test_ids(self):
        self.train_ids = {}
        self.test_ids = {}
        for resolution_scale in self.resolution_scales:
            self.train_ids[resolution_scale] = np.arange(len(self.train_cameras)).tolist()
            self.test_ids[resolution_scale] = np.arange(len(self.test_cameras)).tolist()

    def popTrainCamera(self, scale=1.0):
        current_size = len(self.train_ids[scale])
        if current_size == 0: # reload the ids
            self.reset_train_test_ids()
            current_size = len(self.train_ids[scale])
            if self.args.remove_unseen_points:
                self.gaussians.reset_accum_values_and_remove_low_visibility_points()

        train_id = self.train_ids[scale].pop(randint(0, current_size - 1))
        if not self.args.load_dynamic:
            return self.train_cameras_list[scale][train_id]

        camera = self.train_cameras[train_id]
        return loadCam(self.args, train_id, camera, scale) # read image from file

    def getTestSize(self):
        return len(self.test_cameras)

    # def getTrainCameras(self, scale=1.0):
    #     return self.train_cameras[scale]
    #
    # def getTestCameras(self, scale=1.0):
    #     return self.test_cameras[scale]
