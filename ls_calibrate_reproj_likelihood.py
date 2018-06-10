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

import sys
import shutil
from os.path import isfile
from time import time
import dill as pickle
import numpy as np
from lmfit import Parameters, Minimizer
import emcee
import sympy as sp
from sympy.utilities.autowrap import autowrap, CodeWrapper

from ls_calibrate_base import LSCalibrateBase
from ls_calibrate_core import LSCalibrateCore

pickle.settings['recurse'] = True

point_jac_lambda = None
reproj_err_jac_lambda = None


class LSCalibrateReprojLikelihood(LSCalibrateBase):
    def __init__(self, sensor_size, focal_length, resolution, principal_point, k1, k2, extrinsics, v0,
                 nav_covs, nav_cov_ts, uv_cov, int_cov, dist_cov,
                 ext_rep, nav_rep='euler', ext_cov=None, init_jacs=True,
                 allow_point_jac_compilation=False):
        """
        :param sensor_size: size of sensor in world units (e.g. metres)
        :param focal_length: focal length in world units
        :param resolution: spatial number of pixels per line scan
        :param principal_point: principal point in pixels
        :param k1: first order radial distortion coefficient
        :param k2: second order radial distortion coefficient
        :param extrinsics: position and orientation of sensor.
        :param v0: pixel offset from linescan (for testing only, usually 0)
        :param nav_covs: list of 2D nav variances where each item is a [x_var, y_var, z_var,
        roll_var, pitch_var, yaw_var] vector.
        :param nav_cov_ts: array of timestamps corresponding to each entry in nav_covs
        :param uv_cov: covariance of pixel positions
        :param int_cov: covariance of intrinsics (order: pp, focal length)
        :param dist_cov: covariance of distortion coefficients
        :param ext_rep: extrinsics rotation representation ("aa" or "euler")
        :param nav_rep: nav system solution rotation respresenation ("aa" or "euler")
        :param ext_cov: extrinsics covariance
        :param init_jacs: whether to initialise the jacobian functions
        :param allow_point_jac_compilation: if True, will not throw an exception if wrapper_module_0.so is not found.
        """

        super(LSCalibrateReprojLikelihood, self).__init__(sensor_size=sensor_size,
                                                          focal_length=focal_length,
                                                          resolution=resolution,
                                                          principal_point=principal_point,
                                                          k1=k1, k2=k2,
                                                          extrinsics=extrinsics,
                                                          v0=v0,
                                                          ext_rep=ext_rep, nav_rep=nav_rep)

        self.cov_list = None
        self.nav_covs = nav_covs
        self.nav_cov_ts = nav_cov_ts
        self.uv_cov = uv_cov
        self.int_cov = int_cov
        self.dist_cov = dist_cov
        if ext_cov is None:
            self.ext_cov = np.zeros((6, 6), dtype=float)
        else:
            self.ext_cov = ext_cov
        self.allow_point_jac_compilation = allow_point_jac_compilation

        assert(len(nav_covs) == len(nav_cov_ts))

        self.translation_bound = 2.

        if init_jacs:
            global point_jac_lambda
            point_jac_lambda = self.try_loading_point_jacobian_func('tmp_point_jac.pkl',
                                                                    self.prepare_point_jacobian_func)

            # the reproj error jacobian is quick, so doesn't need to be stored on disk.
            global reproj_err_jac_lambda
            reproj_err_jac_lambda = self.prepare_reproj_err_jac_func()

            self.point_jac_output_shape = self.calc_point_jacobian(0, [0, 0, 0, 0, 0, 0], 0, [0, 0, 0, 0, 0, 0]).shape

        self.point_jac_list = None
        self.reproj_err_jac_list = None

    def try_compiling_point_jacobian_func(self, path, jac_prep_func):
        """
        Attempts to load jacobian function from given location, generate (and save) if does not exist.
        """
        print("Loading point estimate jacobian...", file=sys.stderr)
        if isfile(path):
            jac, args = pickle.load(open(path, 'r'))
            print("... file found, jacobian loaded.", file=sys.stderr)
        else:
            print("... file not found, regenerating jacobian. This may take some time...", file=sys.stderr)
            t = time()
            jac, args = jac_prep_func()
            pickle.dump((jac, args), open(path, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
            print("... done (%f seconds elapsed), saved to file." % (time() - t), file=sys.stderr)

        print("Compiling to fortran...", file=sys.stderr)
        t = time()
        jac_lambda = autowrap(args=tuple(args), expr=jac, tempdir='tmp_dir')
        print("... done (%f seconds elapsed)." % (time() - t), file=sys.stderr)
        # move static object to current folder for later use
        shutil.copy('tmp_dir/wrapper_module_0.so', 'wrapper_module_0.so')
        # delete temporary directory
        shutil.rmtree('tmp_dir')
        return jac_lambda

    def try_loading_point_jacobian_func(self, path, jac_prep_func):
        try:
            import wrapper_module_0
            jac_lambda = wrapper_module_0.autofunc
            CodeWrapper._module_counter += 1
        except ImportError:
            print("Could not load wrapper file, exiting. Make sure wrapper_module_0.so is in the working directory.",
                  file=sys.stderr)

            if self.allow_point_jac_compilation:
                jac_lambda = self.try_compiling_point_jacobian_func(path=path, jac_prep_func=jac_prep_func)
            else:
                raise

        return jac_lambda

    def assign_covariance_matrices(self, max_dist=1.0, verbose=False):
        """
        Generates covariance matrices for each sample based on saved nav variances and sample time stamps. The last
        available variance sample is used for each (closest past sample).
        """
        self.cov_list = list()
        nav_ts = np.array(self.nav_cov_ts)

        # loop through each individual calibration point
        for i, sample_ts_list in enumerate(self.timestamps):
            self.cov_list.append(list())
            # loop through each timestamp for a given calibration point
            for j, sample_ts in enumerate(sample_ts_list):
                if verbose:
                    print("\rProcessing cov matrix for sample %d, instance %d" % (i, j), end=' ', file=sys.stderr)
                diffs = (nav_ts - sample_ts).astype(float)
                min_idx = np.argmin(np.abs(diffs))
                self.cov_list[-1].append(self.nav_covs[min_idx])
                if verbose:
                    print("\n\nNav ts %d matched to sample ts %d (%d us difference)." % \
                                         (nav_ts[min_idx], sample_ts, np.abs(diffs[min_idx])), file=sys.stderr)

                if np.abs(diffs[min_idx]) > max_dist*1e6:
                    print("\nWarning: closest covariance at %d exceeds %f s for sample at %d (%d us)." % \
                                        (nav_ts[min_idx], max_dist, sample_ts, np.abs(diffs[min_idx])), file=sys.stderr)

        if verbose:
            print("", file=sys.stderr)

    def combine_cov_matrices(self, covs):
        """
        Takes any number of covariance matrices and stacks them into one combined covariance matrix
        """
        # remove any None values
        covs = [cov for cov in covs if cov is not None]

        len_covs = np.zeros(len(covs))
        comb_len = 0
        for i, cov in enumerate(covs):
            len_covs[i] = cov.shape[0]
            comb_len += cov.shape[0]

        comb_cov = np.zeros((comb_len, comb_len), dtype=covs[0].dtype)

        this_pos = 0
        for cov, cur_len in zip(covs, len_covs):
            last_pos = this_pos
            this_pos += cur_len
            comb_cov[int(last_pos):int(this_pos), int(last_pos):int(this_pos)] = cov

        return comb_cov

    def prepare_point_jacobian_func(self):
        """
        Uses symbolic calibration class to generate and lambdify a method that calculates the jacobian of the closest
        point on the SECOND line with respect to the two nav solutions for each ray combo, extrinsic parameters,
        intrinsic parameters, and pixel location.
        [u1, v1, nav1, u2, v2, nav2, pp, f, k1, k2, ext]
        (26 values)
        """
        symbolic_lsc = LSCalibrateCore(ext_rep=self.ext_rep, nav_rep=self.nav_rep)
        # data symbols
        u1, v1, x_n1, y_n1, z_n1, theta_x_n1, theta_y_n1, theta_z_n1 = sp.symbols(
            'u_1 v_1 x_n1 y_n1 z_n1 theta_x_n1 theta_y_n1 theta_z_n1')
        u2, v2, x_n2, y_n2, z_n2, theta_x_n2, theta_y_n2, theta_z_n2 = sp.symbols(
            'u_2 v_2 x_n2 y_n2 z_n2 theta_x_n2 theta_y_n2 theta_z_n2')

        nav1 = np.array([x_n1, y_n1, z_n1, theta_x_n1, theta_y_n1, theta_z_n1])
        nav2 = np.array([x_n2, y_n2, z_n2, theta_x_n2, theta_y_n2, theta_z_n2])

        # intrinsics
        f, pp, sensor_size_pix, sensor_size_m = sp.symbols('f u_0 w_p w_m')
        k1, k2 = (None, None) if self.dist_cov is None else sp.symbols('k1 k2')

        # extrinsics
        x_sb, y_sb, z_sb, theta_x_sb, theta_y_sb, theta_z_sb = sp.symbols(
            'x_sb y_sb z_sb theta_xsb theta_ysb theta_zsb')
        ext = np.array([x_sb, y_sb, z_sb, theta_x_sb, theta_y_sb, theta_z_sb])

        # get symbolic jacobian
        jac = symbolic_lsc.get_closest_point_2_jacobian(u1=u1, v1=v1, nav1=nav1,
                                                        u2=u2, v2=v2, nav2=nav2,
                                                        pp=pp, f=f, sensor_size_pix=sensor_size_pix,
                                                        sensor_size_m=sensor_size_m, k1=k1, k2=k2, ext=ext)

        args = [u1] + [v1] + list(nav1) + [u2] + [v2] + list(nav2) + [f] + [pp] + \
               [sensor_size_pix] + [sensor_size_m] + [k1] + [k2] + list(ext)

        args = [arg for arg in args if arg is not None]
        return jac, args

    def calc_point_jacobian(self, u1, nav1, u2, nav2):
        """
        Wrapper for lambdified jacobian (point_jac_lambda) function with more friendly interface, taking into account
        intrinsics and extrinsics saved to object.
        """
        v1 = self.cam_params['v0']
        v2 = self.cam_params['v0']
        args = sum([[u1, v1], list(nav1),
                    [u2, v2], list(nav2),
                    [self.cam_params['focal_length'],
                     self.cam_params['principal_point'],
                     self.resolution, self.sensor_size,
                     self.cam_params['k1'], self.cam_params['k2']],
                    list(self.cam_params['extrinsics'])], [])
        # remove undesired values
        args = [arg for arg in args if arg is not None]
        return point_jac_lambda(*args)

    def calc_point_jacobians(self):
        """
        Calculate point function jacobian for each ray pair, at the current extrinsic and intrinsic
        values.

        Each jac list item has the following format:

        point_jac_list[k][i][j] - jacobian for distinct target point k, sample combination (i, j).

        """
        def calc_ind_point(arg):
            jac_list = list()
            k, (sensor_pixel_list, world_coord_list) = arg
            # loop through instances of this same point/ray
            for i, (u1, nav1) in enumerate(zip(sensor_pixel_list, world_coord_list)):
                # and again because these are ray combinations
                jac_list.append(list())
                for j, (u2, nav2) in enumerate(zip(sensor_pixel_list, world_coord_list)):
                    if i == j:
                        # append zeros, because the jacobian will evaluate to NAN (can't calculate a point for exactly
                        # the same ray...)
                        jac_list[-1].append(np.zeros(self.point_jac_output_shape, dtype=float))
                    else:
                        jac_list[-1].append(self.calc_point_jacobian(u1, nav1, u2, nav2))

            return jac_list

        # loop through distinct calibration points
        self.point_jac_list = [calc_ind_point(arg) for arg in enumerate(zip(self.sensor_pixels, self.world_coords))]

    @staticmethod
    def combine_cov_matrices_vectorized(covs):
        """
        Takes any number of covariance matrices and stacks them into one combined covariance matrix
        Operation is done over last two dimensions, any other dimensions must match
        """
        # remove any None values
        covs = [cov for cov in covs if cov is not None]

        ndims = [cov.ndim for cov in covs]
        cov_lengths = [cov.shape[-1] for cov in covs]
        comb_len = np.sum(cov_lengths)

        other_shape = covs[np.argmax(ndims)].shape[:-2]
        comb_shape = np.concatenate((other_shape, [comb_len], [comb_len]))

        comb_cov = np.zeros(comb_shape)

        this_pos = 0
        for cov, cur_len in zip(covs, cov_lengths):
            last_pos = this_pos
            this_pos += cur_len
            comb_cov[..., last_pos:this_pos, last_pos:this_pos] = cov

        return comb_cov

    def calc_point_covariances(self):
        """
        Calculate uncertainties based nav system variance and currently set extrinsic variance.
        """
        self.calc_point_jacobians()

        # loop through jacobians and covariances for individual distinct calibration points
        cov_list = np.array(self.cov_list)
        point_cov_list = list()
        for k, (jacs, covs) in enumerate(zip(self.point_jac_list, cov_list)):
            comb_covs = self.combine_cov_matrices_vectorized((self.uv_cov,
                                                              np.repeat(covs[:, None, :, :], repeats=covs.shape[0], axis=1),
                                                              self.uv_cov, covs, self.int_cov, self.dist_cov,
                                                              self.ext_cov))
            jacs = np.array(jacs)
            # jacs shape is (num_obs, num_obs, num_outputs, num_inputs)
            # covs shape is (num_obs, num_obs, num_inputs, num_inputs)
            point_cov = np.einsum('xyki,xyij->xykj', jacs, comb_covs)
            point_cov_list.append(np.einsum('xyji,xyki->xyjk', point_cov, jacs))

        return point_cov_list

    def calc_weighted_avg_points(self):
        """
        Calculates weighted average of all points
        """
        # list of length k, n x n x 3 each.
        point_list = self.calc_closest_points_2()

        # covariances for each closest point line location
        point_covs = self.calc_point_covariances()

        # loop through each independent point
        avg_cov_list = list()
        avg_point_list = list()

        for k, (points, covs) in enumerate(zip(point_list, point_covs)):
            covs = np.array(covs)

            # set all i = j to I*np.inf. This will cause the inversion to be
            # nan, which will be turned to zero by nan_to_num below
            for i in range(len(covs)):
                covs[i, i] = np.eye(covs.shape[-1]) * np.inf

            denominator_array = np.linalg.inv(covs)
            numerator_array = np.einsum('xyij,xyi->xyj', denominator_array, points)

            avg_cov_list.append(np.linalg.inv(np.sum(np.nan_to_num(denominator_array), (0, 1))))
            avg_point_list.append(avg_cov_list[-1].dot(np.sum(np.nan_to_num(numerator_array), (0, 1))))

        return avg_point_list, avg_cov_list

    def prepare_reproj_err_jac_func(self):
        """
        Uses symbolic calibration class to generate and lambdify a method that calculates the jacobian of the
        reprojection line with respect to the pixel location, current nav solution, feature point, intrinsics,
        and extrinsic parameters
        (21 values).
        Pattern: p, u, v, nav, pp, f, k1, k2, ext
        """
        symbolic_lsc = LSCalibrateCore(ext_rep=self.ext_rep, nav_rep=self.nav_rep)

        # data symbols
        # feature point
        p_x, p_y, p_z = sp.symbols('p_x p_y p_z')
        p = np.array([p_x, p_y, p_z])

        # nav
        u, v, x_n, y_n, z_n, theta_x_n, theta_y_n, theta_z_n = sp.symbols(
            'u, v, x_n y_n z_n theta_x_n theta_y_n theta_z_n')

        nav = np.array([x_n, y_n, z_n, theta_x_n, theta_y_n, theta_z_n])

        # intrinsics
        f, pp, sensor_size_pix, sensor_size_m = sp.symbols('f u_0 w_p w_m')
        # distortion values are optional
        k1, k2 = (None, None) if self.dist_cov is None else sp.symbols('k1 k2')

        # extrinsics
        x_sb, y_sb, z_sb, theta_x_sb, theta_y_sb, theta_z_sb = sp.symbols(
            'x_sb y_sb z_sb theta_xsb theta_ysb theta_zsb')
        ext = np.array([x_sb, y_sb, z_sb, theta_x_sb, theta_y_sb, theta_z_sb])

        # get symbolic jacobian
        jac = symbolic_lsc.get_reproj_err_jacobian(p, u, v, nav, pp, f, sensor_size_pix, sensor_size_m, k1, k2, ext)

        # lambdify and return
        args = [p_x, p_y, p_z, u, v, x_n, y_n, z_n, theta_x_n, theta_y_n, theta_z_n,
                f, pp, sensor_size_pix, sensor_size_m, k1, k2,
                x_sb, y_sb, z_sb, theta_x_sb, theta_y_sb, theta_z_sb]
        # remove any unwanted parameters (e.g. if distortion is not supplied)
        args = [arg for arg in args if arg is not None]
        return autowrap(args=tuple(args), expr=jac)

    def calc_reproj_err_jacobian(self, p, u, nav):
        """
        Wrapper for lamdified jacobian (self.reproj_jac_lambda) function with more friendly interface, taking into
        account intrinsics and extrinsics saved to object.
        """
        v = self.cam_params['v0']
        args = sum([list(p), [u, v], list(nav),
                    [self.cam_params['focal_length'], self.cam_params['principal_point'],
                     self.resolution, self.sensor_size,
                     self.cam_params['k1'], self.cam_params['k2']],
                    list(self.cam_params['extrinsics'])], [])
        # remove undesired parameters (e.g. if distortion is None...)
        args = [arg for arg in args if arg is not None]
        return reproj_err_jac_lambda(*args)

    def calc_reproj_err_jacobians(self, p_est):
        """
        Calculate reprojection error jacobian for each nav position.

        Each jac list item has the following format:

        reproj_err_jac_list[k][i] - jacobian for distinct target point k, nav position i

        """
        def calc_ind_point(arg):
            jac_list = list()
            k, (sensor_pixel_list, world_coord_list, p) = arg
            # loop through instances of this same point/ray
            for i, (u, nav) in enumerate(zip(sensor_pixel_list, world_coord_list)):
                # and again because these are ray combinations
                jac_list.append(self.calc_reproj_err_jacobian(p, u, nav))

            return jac_list

        # loop through distinct calibration points
        self.reproj_err_jac_list = [calc_ind_point(arg) for arg in
                                    enumerate(zip(self.sensor_pixels, self.world_coords, p_est))]

    def calc_reproj_err_covariances(self, points, p_covs):
        """
        Calculate uncertainties based nav system variance and currently set extrinsic variance.

        points is a list of feature points
        p_cov is a list of covariances for the feature point position.
        """
        self.calc_reproj_err_jacobians(points)

        cov_list = list()

        # loop through jacobians and covariances for individual distinct calibration points
        for k, (jacs, covs, p_cov) in enumerate(zip(self.reproj_err_jac_list, self.cov_list, p_covs)):
            cov_list.append(list())
            for i, (jac, nav_cov) in enumerate(zip(jacs, covs)):
                # at this point we have the jacobian (15 length vector p, nav, ext), covariance matrices for
                # p and nav solutions (6x6 each)
                # need to generate matching covariance matrix (21 x 21 max), assume zeros for ext.
                # p, u, nav, pp, f, ext
                comb_cov = self.combine_cov_matrices((p_cov, self.uv_cov, nav_cov,
                                                      self.int_cov, self.dist_cov, self.ext_cov))
                cov_list[-1].append(np.asscalar(jac.dot(comb_cov).dot(jac.T)))

        return cov_list

    def log_likelihood(self, params, extrinsic_mapping):
        """
        Calculates log likelihood of parameters given the data.
        """
        # update current parameters
        for param, mapping in zip(params, extrinsic_mapping):
            self.cam_params['extrinsics'][mapping] = param

        # intrinsics are optional
        self.cam_params['focal_length'] = self.cam_params['focal_length'] if len(params) < 7 else params[6]
        self.cam_params['principal_point'] = self.cam_params['principal_point'] if len(params) < 8 else params[7]
        self.cam_params['k1'] = self.cam_params['k1'] if len(params) < 9 else params[8]
        self.cam_params['k2'] = self.cam_params['k2'] if len(params) < 10 else params[9]

        # update all rays
        self.calc_all_ray_points()

        # calculate avg points
        self.point_estimates, avg_cov_list = self.calc_weighted_avg_points()

        reproj_err_vars = np.array(self.calc_reproj_err_covariances(self.point_estimates, avg_cov_list))

        # calculate all reprojection errors
        sq_errors = np.array(self.calc_reprojection_sq_errors())

        return -0.5 * np.sum(sq_errors / reproj_err_vars)


    def objective_func(self, params, objective_list, extrinsic_mapping, intrinsic_params=None):
        """
        Objective function to be called by lmfit based optimisers
        """

        x = [params['extrinsics0'].value, params['extrinsics1'].value, params['extrinsics2'].value,
             params['extrinsics3'].value, params['extrinsics4'].value, params['extrinsics5'].value]

        for key in intrinsic_params:
            x += [params[key].value]

        log_l = self.log_likelihood(np.array(x), extrinsic_mapping=extrinsic_mapping)

        residual = -log_l

        objective_list.append(residual)

        return residual

    def optimise_params(self, extrinsics=True, intrinsics=None, method=None,
                        minimizer_kws=None):
        """
        Initialise lmfit based optimisation.
        """
        objective_list = list()
        intrinsics = [] if intrinsics is None else intrinsics

        # create a set of parameters
        params = Parameters()
        for key in list(self.cam_params.keys()):
            if key == "extrinsics":
                for ext_id, ext_val in enumerate(self.cam_params[key]):
                    if self.ext_rep == 'aa' and ext_id >= 3:
                        min_val = -np.pi
                        max_val = np.pi
                    else:
                        min_val = -np.inf
                        max_val = np.inf

                    if ext_id < 3:
                        min_val = ext_val - self.translation_bound
                        max_val = ext_val + self.translation_bound

                    if extrinsics:
                        params.add(key + str(ext_id), value=ext_val, vary=True, min=min_val, max=max_val)
                    else:
                        params.add(key + str(ext_id), value=ext_val, vary=False)

            else:
                # add non-extrinsic key, set to non varying
                params.add(key, value=self.cam_params[key], vary=False)
                if key in intrinsics:
                    params[key].set(vary=True)

        # Extrinsic mapping is really only for uncertainty estimator at this point. The following selects all
        # extrinsics.
        extrinsic_mapping = np.arange(6)

        args = {} if minimizer_kws is None else minimizer_kws

        mini = Minimizer(self.objective_func, params,
                         fcn_args=(objective_list, extrinsic_mapping, intrinsics), **args)
        result = mini.minimize(method=method)

        self.last_min_obj = mini
        self.last_result = result

        # update parameters one final time by running objective function (passing dummy list)
        self.objective_func(result.params, list(), extrinsic_mapping, intrinsic_params=intrinsics)

        return objective_list

    def optimise(self, intrinsics=None, method='powell', minimizer_kws=None):
        print("\nAssigning covariance matrices to each sample...", file=sys.stderr)
        self.assign_covariance_matrices()
        objective = self.optimise_params(intrinsics=intrinsics,
                                         extrinsics=True, method=method,
                                         minimizer_kws=minimizer_kws)
        print("Final residual: %f" % objective[-1], file=sys.stderr)

        return objective

    def mcmc_sampling(self, starting_cov, num_walkers, num_samples_per_walker, log_file,
                                  burnin_samples=100, mcmc_scale=2.0,
                                  include_intrinsics=0, extrinsic_mapping=None):

        # extrinsic mapping allows us to fix certain extrinsic variables, if they are not listed
        extrinsic_mapping = np.arange(6) if extrinsic_mapping is None else extrinsic_mapping

        p_start = self.cam_params['extrinsics']
        p_start = p_start[extrinsic_mapping]

        # intrinsics are optional
        p_start = p_start if include_intrinsics < 1 else np.append(p_start, self.cam_params['focal_length'])
        p_start = p_start if include_intrinsics < 2 else np.append(p_start, self.cam_params['principal_point'])
        p_start = p_start if include_intrinsics < 3 else np.append(p_start, self.cam_params['k1'])
        p_start = p_start if include_intrinsics < 4 else np.append(p_start, self.cam_params['k2'])

        ndim = len(p_start)
        # starting cov must match the size of extrinsic_mapping
        p0 = np.random.multivariate_normal(p_start, starting_cov, num_walkers)
        sampler = emcee.EnsembleSampler(num_walkers, ndim, self.log_likelihood, args=[extrinsic_mapping],
                                        a=mcmc_scale)

        print("\nAssigning covariance matrices to each sample...", file=sys.stderr)
        self.assign_covariance_matrices()

        print("Running MCMC burn in with %d walkers and %d samples (scale parameter = %.2f)..." % (num_walkers,
                                                                                                   burnin_samples,
                                                                                                   mcmc_scale),
              file=sys.stderr)

        pos, prob, state = sampler.run_mcmc(p0, burnin_samples)
        sampler.reset()

        print("Running main MCMC sampler with %d walkers and %d samples each (scale parameter = %.2f)..." % (num_walkers,
                                                                                                             num_samples_per_walker,
                                                                                                             mcmc_scale),
              file=sys.stderr)

        sampler.run_mcmc(pos, num_samples_per_walker)

        if log_file is not None:
            log = {'samples': sampler.flatchain,
                   'log_L_ext': sampler.flatlnprobability,
                   'chain': sampler.chain}
            print("\nSaving log to %s." % log_file, file=sys.stderr)
            with open(log_file, 'w') as f:
                pickle.dump(log, f, protocol=pickle.HIGHEST_PROTOCOL)

        return sampler.chain, sampler.lnprobability, sampler.acceptance_fraction
