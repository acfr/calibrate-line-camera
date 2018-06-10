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
import sympy as sp


class PointManipulate3D(object):
    def __init__(self, symbolic=False):
        self.test = None
        self.dtype = object if symbolic else float
        self.np = sp if symbolic else np
        # because numpy and sympy names arc trig functions are different...
        self.arcsin = sp.asin if symbolic else np.arcsin
        self.arctan2 = sp.atan2 if symbolic else np.arctan2

    def make_T(self, p):
        return np.array([[1., 0., 0., p[0]], [0., 1., 0., p[1]], [0., 0., 1., p[2]], [0., 0., 0., 1.]], dtype=self.dtype)

    def make_R_from_euler(self, r, order):
        roll, pitch, yaw = r[0], r[1], r[2]
        rx = np.array([[1, 0, 0], [0, self.np.cos(roll), -self.np.sin(roll)], [0, self.np.sin(roll), self.np.cos(roll)]])
        ry = np.array([[self.np.cos(pitch), 0, self.np.sin(pitch)], [0, 1, 0], [-self.np.sin(pitch), 0, self.np.cos(pitch)]])
        rz = np.array([[self.np.cos(yaw), -self.np.sin(yaw), 0], [self.np.sin(yaw), self.np.cos(yaw), 0], [0, 0, 1]])
        if order == 'xyz':
            R = np.pad(rx.dot(ry).dot(rz), ((0, 1), (0, 1)), mode='constant', constant_values=(0., 0.))
        elif order == 'zyx':
            R = np.pad(rz.dot(ry).dot(rx), ((0, 1), (0, 1)), mode='constant', constant_values=(0., 0.))
        else:
            raise ValueError("Order '%s' not supported." % order)
        R[3, 3] = 1.
        return R

    def make_R_from_aa(self, aa):
        theta, n1, n2, n3 = aa
        S_n = np.array([[0, -n3, n2],
                        [n3, 0, -n1],
                        [-n2, n1, 0]])
        R = np.eye(3) + self.np.sin(theta) * S_n + (1 - self.np.cos(theta)) * S_n.dot(S_n)
        return np.concatenate((np.concatenate((R, np.zeros((3, 1), dtype=float)), axis=1),
                               np.array([[0., 0., 0., 1.]])), axis=0)

    def make_R(self, r, rep, inv=False):
        if rep == 'aa':
            aa = self.check_aa_dims(r)
            R = self.make_R_from_aa(aa*np.array([-1., 1., 1., 1.]) if inv else aa)
        elif rep == 'euler':
            if inv:
                R = self.make_R_from_euler(-r, order='xyz')
            else:
                R = self.make_R_from_euler(r, order='zyx')
        else:
            raise ValueError("'%s' representation not supported." % rep)
        return R

    def points_frame_to(self, p, ref, rep):
        T = self.make_T(-ref[:3])
        R = self.make_R(ref[3:], rep, inv=True)
        RT = R.dot(T)
        p_new = RT.dot(np.append(p, 1.))
        p_new /= p_new[3]
        return p_new[:3]

    def points_frame_from(self, p, ref, rep):
        T = self.make_T(ref[:3])
        R = self.make_R(ref[3:], rep)
        RT = T.dot(R)
        p_new = RT.dot(np.append(p, 1.))
        p_new /= p_new[3]
        return p_new[:3]

    def check_aa_dims(self, aa):
        if len(aa) == 4:
            return aa
        elif len(aa) == 3:
            theta = self.np.sqrt(np.sum(aa**2))
            n1, n2, n3 = aa / theta
            return np.array([theta, n1, n2, n3])
        else:
            raise ValueError("Axis-angle vector must be 3 or 4 dimensional.")

    def euler_to_quat(self, euler):
        x, y, z = euler
        q = np.empty(4, self.dtype)
        q[0] = self.np.cos(z / 2) * self.np.cos(y / 2) * self.np.cos(x / 2) + self.np.sin(z / 2) * self.np.sin(y / 2) * self.np.sin(x / 2)
        q[1] = self.np.cos(z / 2) * self.np.cos(y / 2) * self.np.sin(x / 2) - self.np.sin(z / 2) * self.np.sin(y / 2) * self.np.cos(x / 2)
        q[2] = self.np.cos(z / 2) * self.np.sin(y / 2) * self.np.cos(x / 2) + self.np.sin(z / 2) * self.np.cos(y / 2) * self.np.sin(x / 2)
        q[3] = self.np.sin(z / 2) * self.np.cos(y / 2) * self.np.cos(x / 2) - self.np.cos(z / 2) * self.np.sin(y / 2) * self.np.sin(x / 2)
        return q

    def quat_to_axis_angle(self, quat):
        assert(len(quat) == 4)
        e = quat[1:]
        e0 = quat[0]
        axis_angle = np.empty(4, self.dtype)
        numerator = self.np.sqrt(1 - e0**2)
        axis_angle[1:] = e / numerator
        axis_angle[0] = 2 * self.arctan2(numerator, e0)
        return axis_angle

    def axis_angle_to_quat(self, axis_angle):
        assert (len(axis_angle) == 4)
        n = axis_angle[1:]
        theta = axis_angle[0]
        e = self.np.sin(theta/2) * n
        e0 = self.np.cos(theta/2)
        return np.insert(e, 0, e0)

    def quat_to_euler(self, quat):
        assert (len(quat) == 4)
        e0, e1, e2, e3 = quat
        return np.array([self.arctan2((e2*e3 + e0*e1), 0.5-(e1**2 + e2**2)),
                         self.arcsin(-2*(e1*e3 - e0*e2)),           # returns [-pi/2, pi/2] by default as required.
                         self.arctan2((e1*e2 + e0*e3), 0.5-(e2**2 + e3**2))])

    def euler_to_axis_angle(self, euler):
        quat = self.euler_to_quat(euler)
        return self.quat_to_axis_angle(quat)

    def axis_angle_to_euler(self, axis_angle):
        assert (len(axis_angle) == 4)
        quat = self.axis_angle_to_quat(axis_angle)
        return self.quat_to_euler(quat)