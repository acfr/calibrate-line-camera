#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This file is part of calibrate-line-camera
# Copyright (c) 2011 The University of Sydney
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the University of Sydney nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
# GRANTED BY THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
# HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import numpy as np

from point_manipulation import PointManipulate3D
from ls_calibrate_core import LSCalibrateCore


class LSCalibrateBase(LSCalibrateCore):
    def __init__(self, sensor_size, focal_length, resolution, principal_point, k1, k2, extrinsics, v0,
                 ext_rep, nav_rep='euler'):
        super(LSCalibrateBase, self).__init__(symbolic=False, ext_rep=ext_rep, nav_rep=nav_rep)
        self.sensor_size = sensor_size
        self.resolution = resolution

        self.cam_params = {'extrinsics': extrinsics,
                           'focal_length': focal_length,
                           'principal_point': principal_point,
                           'k1': k1,
                           'k2': k2,
                           'v0': v0}

        self.RT_cam = None
        self.sensor_pixels = list()
        self.sensor_points = list()
        self.target_point_dists = None
        self.world_coords = list()
        self.timestamps = list()
        self.ray_points = list()
        self.point_estimates = None
        self.reprojections = None
        self.last_min_obj = None
        self.last_result = None
        self.target_points = None
        self.p3d = PointManipulate3D()

        self.tol = 1e-6

    def add_data_point(self, ts, x, world_coords, index=0):
        """
        Add a single data point.
        """
        if index + 1 > len(self.sensor_pixels):
            self.sensor_pixels.append(list())
            self.world_coords.append(list())
            self.timestamps.append(list())
        self.sensor_pixels[index].append(float(x))
        self.world_coords[index].append(world_coords)
        self.timestamps[index].append(ts)

    def set_cam_trans_rot_matrices(self, ext=None):
        """
        Sets camera translation matrix for efficiency (to be used in cam_to_world_coords)
        """
        ext = self.cam_params['extrinsics']
        self.RT_cam = super(LSCalibrateBase, self).set_cam_trans_rot_matrices(ext=ext)

    def image_to_pinhole(self, u, v=None, ppx=None, f=None, sensor_size_pix=None,
                         sensor_size_m=None, k1=None, k2=None):
        """
        Takes pixel on sensor and returns returns actual point location
        relative to camera frame.
        """
        return super(LSCalibrateBase, self).image_to_pinhole(u=u, v=self.cam_params['v0'],
                                                             ppx=self.cam_params['principal_point'],
                                                             f=self.cam_params['focal_length'],
                                                             sensor_size_pix=self.resolution,
                                                             sensor_size_m=self.sensor_size,
                                                             k1=self.cam_params['k1'],
                                                             k2=self.cam_params['k2'])

    def pinhole_to_image(self, px, py, pz, ppx=None, f=None, sensor_size_pix=None,
                         sensor_size_m=None, k1=None, k2=None):
        """
        Takes point in image coordinates and converts it to pixel coordinates, including distortion.
        """
        return super(LSCalibrateBase, self).pinhole_to_image(px=px, py=py, pz=pz,
                                                             ppx=self.cam_params['principal_point'],
                                                             f=self.cam_params['focal_length'],
                                                             sensor_size_pix=self.resolution,
                                                             sensor_size_m=self.sensor_size,
                                                             k1=self.cam_params['k1'],
                                                             k2=self.cam_params['k2'])

    def cam_to_world_coords(self, p, nav, ext=None):
        """
        Transforms x,y,z point from camera from to world frame.
        """
        # transform and rotate from camera frame first
        p = self.RT_cam.dot(np.append(p, 1))
        p /= p[3]
        # then transform and rotate from robot frame
        return self.p3d.points_frame_from(p[:3], nav, self.nav_rep)

    def world_to_cam_coords(self, p, nav, ext=None):
        """
        Transforms x,y,z point from world to camera frame.
        """
        ext = self.cam_params['extrinsics']
        return super(LSCalibrateBase, self).world_to_cam_coords(p=p, nav=nav, ext=ext)

    def calc_all_ray_points(self, ray_scale=1.0):
        """
        Calculates two points in world coordinates on each ray, based on the current parameters.
        """
        self.set_cam_trans_rot_matrices()
        self.ray_points = list()
        # loop through distinct calibration points
        for sensor_pixel_list, world_coord_list in zip(self.sensor_pixels,
                                                       self.world_coords):
            # loop through instances of this same point
            self.ray_points.append(list())
            for sensor_pixel, world_coord in zip(sensor_pixel_list, world_coord_list):
                ps = np.array(self.image_to_pinhole(sensor_pixel)) * ray_scale
                ps = self.cam_to_world_coords(ps, world_coord)
                pc = self.cam_to_world_coords(np.array([0.0, 0.0, 0.0]), world_coord)
                self.ray_points[-1].append((ps, pc))

    def calc_closest_points_2(self):
        """
        Calculate closest points on all rays (for SECOND ray on each combination).
        """
        points_list = list()
        for ray_points in self.ray_points:
            ray_points = np.array(ray_points)

            vectors = ray_points[:, 0] - ray_points[:, 1]

            # calculate unit vector perpendicular to line pairs
            d1 = np.repeat(vectors, vectors.shape[0], 0)
            d2 = np.tile(vectors, (vectors.shape[0], 1))
            cross = np.cross(d1, d2)
            with np.errstate(divide='ignore', invalid='ignore'):
                n = cross / np.expand_dims(np.linalg.norm(cross, ord=None, axis=1), 1)

            # create array of points on line pairs
            p1 = np.repeat(ray_points[:, 0], ray_points.shape[0], 0)
            p2 = np.tile(ray_points[:, 0], (ray_points.shape[0], 1))

            # when rays are crossed with themselves, a NAN results - set to 0.
            n[np.isnan(n)] = 0

            # calculate all closest points
            n1 = np.cross(d1, n)
            with np.errstate(divide='ignore', invalid='ignore'):
                c2 = p2 + np.expand_dims(np.einsum("ij,ij->i", (p1 - p2), n1) / np.einsum("ij,ij->i", d2, n1), 1) * d2

            # at this point we have an (n x n) x 3 array. Need to split into n x n x 3 array.
            points_list.append(np.reshape(c2, (vectors.shape[0], vectors.shape[0], 3)))

        return points_list

    def reproject_point(self, p, nav, f=None, ext=None):
        """
        Reproject point to sensor x,y location based on current intrinsics and extrinsics.
        """
        ext = self.cam_params['extrinsics']
        return super(LSCalibrateBase, self).reproject_point(p=p, nav=nav, f=self.cam_params['focal_length'], ext=ext)

    def calc_reprojection_sq_errors(self):
        """
        Calculate reprojection error for all points and viewing locations.
        """
        self.set_cam_trans_rot_matrices()
        to_return = list()
        # loop through distinct calibration points
        for point_est, sensor_pixel_list, world_coord_list in zip(self.point_estimates,
                                                                  self.sensor_pixels,
                                                                  self.world_coords):
            # loop through instances of this same point
            to_return.append(list())
            for sensor_pixel, world_coord in zip(sensor_pixel_list, world_coord_list):
                ps = np.array(self.image_to_pinhole(sensor_pixel))
                point_est_cam = self.reproject_point(point_est, world_coord)
                to_return[-1].append(np.sum((ps - point_est_cam) ** 2))

        return np.array(to_return)
