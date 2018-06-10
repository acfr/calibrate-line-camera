#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This file is part of calibrate-line-scanning-camera
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
import sympy as sp

from point_manipulation import PointManipulate3D


class LSCalibrateCore(object):
    def __init__(self, ext_rep, nav_rep, symbolic=True):
        self.symbolic = symbolic
        self.p3d = PointManipulate3D(symbolic=symbolic)
        self.ext_rep = ext_rep
        self.nav_rep = nav_rep

    def image_to_pinhole(self, u, v, ppx, f, sensor_size_pix, sensor_size_m, k1, k2):
        """
        Takes pixel on sensor and returns returns actual point location
        relative to camera frame in normalised coordinates (z = 1.0)
        """
        f_pix = (f / sensor_size_m * sensor_size_pix)
        normalised_x = (u - ppx) / f_pix
        normalised_y = v / f_pix
        # do not undistort if no k1 or k2 values are given
        normalised_x = normalised_x if (k1 is None) or (k2 is None) else self.undistort_pixel(normalised_x, k1, k2)
        # No undistortion is applied to y in the 1D case. The y location only accounts for the uncertainty of the actual
        # y location in the world. ppy is assumed to be 0.
        return normalised_x, normalised_y, 1.0

    def pinhole_to_image(self, px, py, pz, ppx, f, sensor_size_pix, sensor_size_m, k1, k2):
        """
        Takes point in image coordinates and converts it to pixel coordinates, including distortion.
        """
        pix_per_m = sensor_size_pix / sensor_size_m
        normalised_x = self.distort_pixel(px / pz, k1, k2)
        normalised_y = py / pz
        u = normalised_x * f * pix_per_m + ppx
        v = normalised_y * f * pix_per_m
        return u, v

    @staticmethod
    def distort_pixel(x, k1, k2):
        """
        Distorts pixel according to "An Analytical Piecewise Radial Distortion Model for Precision Camera Calibration"
        """
        return x * (1 + k1*x**2 + k2*x**4)

    @staticmethod
    def undistort_pixel(x, k1, k2):
        """
        Adjusts for radial distortion, given x normalised point location (where x=0 is at principal point).

        Approximation as described in "An Analytical Piecewise Radial Distortion Model for Precision Camera Calibration"
        Maybe implement different piecewise model later (for exact solution, but k values don't match opencv...)
        """
        r = x * (1 - k1*x**2 - k2*x**4)
        return x / (1 + k1 * r ** 2 + k2 * r ** 4)

    def set_cam_trans_rot_matrices(self, ext):
        """
        Sets camera translation matrix for efficiency (to be used in cam_to_world_coords)
        """
        R = self.p3d.make_R(ext[3:], rep=self.ext_rep)
        T = self.p3d.make_T(ext[:3])
        return T.dot(R)

    def cam_to_world_coords(self, p, nav, ext):
        """
        Transforms x,y,z point from camera to world frame.
        """
        RT_cam = self.set_cam_trans_rot_matrices(ext)

        # transform and rotate from camera frame first
        p = RT_cam.dot(np.append(p, 1))
        p /= p[3]
        # then transform and rotate from robot frame
        return self.p3d.points_frame_from(p[:3], nav, self.nav_rep)

    def world_to_cam_coords(self, p, nav, ext):
        """
        Transforms x,y,z point from world to camera frame.
        """
        # transform and rotate from world to robot frame
        p = self.p3d.points_frame_to(p, nav, rep=self.nav_rep)
        # then transform and rotate to camera frame
        return self.p3d.points_frame_to(p, np.array(ext), rep=self.ext_rep)

    def calc_ray_points(self, u, v, nav, pp, f, sensor_size_pix, sensor_size_m, k1, k2, ext):
        """
        Calculates two points for single ray as given by arguments.
        """
        ps = np.array(self.image_to_pinhole(u=u, v=v, ppx=pp, f=f, sensor_size_pix=sensor_size_pix,
                                            sensor_size_m=sensor_size_m, k1=k1, k2=k2))
        ps = self.cam_to_world_coords(ps, nav, ext)
        pc = self.cam_to_world_coords(np.array([0, 0, 0]), nav, ext)
        return ps, pc

    def calc_closest_point_2(self, u1, v1, nav1, u2, v2, nav2,
                             pp, f, sensor_size_pix, sensor_size_m, k1, k2, ext):
        """
        Calculate symbolic solution of closest point on the SECOND line segment.
        """
        ps1, pc1 = self.calc_ray_points(u=u1, v=v1, nav=nav1, pp=pp, f=f, sensor_size_pix=sensor_size_pix,
                                        sensor_size_m=sensor_size_m, k1=k1, k2=k2, ext=ext)
        ps2, pc2 = self.calc_ray_points(u=u2, v=v2, nav=nav2, pp=pp, f=f, sensor_size_pix=sensor_size_pix,
                                        sensor_size_m=sensor_size_m, k1=k1, k2=k2, ext=ext)

        # calculate direction vectors
        v1 = ps1 - pc1
        v2 = ps2 - pc2

        # calculate vector perpendicular to line pairs
        n = np.cross(v1, v2)

        n1 = np.cross(v1, n)

        return sp.Matrix(ps2 + ((ps1 - ps2).dot(n1) / v2.dot(n1)) * v2)

    def reproject_point(self, p, nav, f, ext):
        """
        Reproject point to sensor x,y location based on current intrinsics and extrinsics.
        """
        p = self.world_to_cam_coords(p=p, nav=nav, ext=ext)
        return [p[0]/p[2], p[1]/p[2], 1.0]

    def reproj_err(self, p, u, v, nav, pp, f, sensor_size_pix, sensor_size_m, k1, k2, ext):
        """
        Calculate symbolic solution for reprojection error, given current nav, feature point, intrinsics and
        extrinsics.
        """
        p_reproj = np.array(self.reproject_point(p=p, nav=nav, f=f, ext=ext))
        p_meas = np.array(self.image_to_pinhole(u=u, v=v, ppx=pp, f=f, sensor_size_pix=sensor_size_pix,
                                                sensor_size_m=sensor_size_m, k1=k1, k2=k2))

        return sp.Matrix([sp.sqrt(np.sum((p_reproj - p_meas)**2))])

    def get_closest_point_2_jacobian(self, u1, v1, nav1, u2, v2, nav2,
                                     pp, f, sensor_size_pix, sensor_size_m, k1, k2, ext):
        """
        Calculate symbolic jacobian solution of closest point on the SECOND line segment between two lines,
        wrt the two nav solutions for each ray combo, pixel locations, intrinsic parameters,
        extrinsic parameters (22 items total)
        """
        p = self.calc_closest_point_2(u1=u1, v1=v1, nav1=nav1, u2=u2, v2=v2, nav2=nav2,
                                      pp=pp, f=f, sensor_size_pix=sensor_size_pix, sensor_size_m=sensor_size_m,
                                      k1=k1, k2=k2, ext=ext)
        inputs = np.concatenate(([u1], [v1], nav1, [u2], [v2], nav2, [pp], [f], [k1], [k2], ext))
        # remove any None values (any parameters that are not desired)
        inputs = sp.Matrix(inputs[inputs != np.array(None)])
        return p.jacobian(inputs)

    def get_reproj_err_jacobian(self, p, u, v, nav, pp, f, sensor_size_pix, sensor_size_m, k1, k2, ext):
        """
        Calculate symbolic jacobian solution of reprojection error, given current nav, pixel loc, feature point,
        intrinsics and extrinsics.
        """
        d = self.reproj_err(p=p, u=u, v=v, nav=nav, pp=pp, f=f, sensor_size_pix=sensor_size_pix,
                            sensor_size_m=sensor_size_m, k1=k1, k2=k2, ext=ext)
        inputs = np.concatenate((p, [u], [v], nav, [pp], [f], [k1], [k2], ext))
        # remove any None values (any parameters that are not desired)
        inputs = sp.Matrix(inputs[inputs != np.array(None)])
        return d.jacobian(inputs)

    def get_projection_jacobian(self, u, v, nav, pp, f, sensor_size_pix, sensor_size_m, k1, k2, ext, plane_params):
        """
        Calculate symbolic jacobian solution of a projected point to a plane, given current nav, pixel loc, feature
        point, intrinsics and extrinsics.
        """
        ps, pc = self.calc_ray_points(u=u, v=v, nav=nav, pp=pp, f=f, sensor_size_pix=sensor_size_pix,
                                      sensor_size_m=sensor_size_m, k1=k1, k2=k2, ext=ext)
        p_int = self.project_ray_to_plane(rp1=ps, rp2=pc, plane_params=plane_params)
        inputs = np.concatenate(([u], [v], nav, [pp], [f], [k1], [k2], ext))
        inputs = sp.Matrix(inputs[inputs != np.array(None)])
        return sp.Matrix(p_int).jacobian(inputs)

    def transform_axis_angle_to_euler(self, trans):
        """
        Convert 6 DoF axis angle representation to euler.
        """
        assert(len(trans) == 6)
        theta = self.p3d.np.sqrt(np.sum(trans[3:] ** 2))
        n = trans[3:] / theta
        euler_trans = np.copy(trans)
        euler_trans[3:] = self.p3d.axis_angle_to_euler(np.insert(n, 0, theta))
        return euler_trans
