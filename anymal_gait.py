import numpy as np

class Leg:
    def __init__(self, name, side=1, l0=0.0585, l1=0.35, l2=0.33):
        self.name = name
        self.side = side 
        self.l0, self.l1, self.l2 = l0, l1, l2
        self.q = np.zeros(3)

    def fk(self, q=None):
        if q is not None:
            self.q = np.asarray(q, float)
        q1, q2, q3 = self.q

        x =  self.l1 * np.sin(q2)          + self.l2 * np.sin(q2 + q3)
        y =  self.side * self.l0 * np.cos(q1)
        z = -self.l1 * np.cos(q2)          - self.l2 * np.cos(q2 + q3)

        return np.array([x, y, z])

    def ik(self, p):
        x, y, z = p

        ryz2 = y**2 + z**2 - self.l0**2
        ryz  = np.sqrt(max(ryz2, 1e-9))
        q1   = np.arctan2(y, -z) - np.arctan2(self.side * self.l0, ryz)

        D  = (x**2 + z**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        q3 = -np.arccos(np.clip(D, -1, 1))

        alpha = np.arctan2(x, -z)
        beta  = np.arctan2(self.l2 * np.sin(-q3), self.l1 + self.l2 * np.cos(q3))
        q2    = alpha - beta

        return np.array([q1, q2, q3])

    def jacobian(self, q=None):
        if q is None:
            q = self.q
        q1, q2, q3 = q
        J = np.zeros((3, 3))

        J[0, 1] =  self.l1 * np.cos(q2) + self.l2 * np.cos(q2 + q3)
        J[0, 2] =  self.l2 * np.cos(q2 + q3)

        J[1, 0] = -self.side * self.l0 * np.sin(q1)

        J[2, 1] =  self.l1 * np.sin(q2) + self.l2 * np.sin(q2 + q3)
        J[2, 2] =  self.l2 * np.sin(q2 + q3)

        return J

    def det_J(self, q=None):
        return np.linalg.det(self.jacobian(q))

    def is_singular(self, q=None, tol=1e-3):
        return abs(self.det_J(q)) < tol


class ANYmal:
    LEGS = ['LF', 'RF', 'LH', 'RH']

    def __init__(self):
        self.legs = {
            'LF': Leg('LF', side=+1),
            'RF': Leg('RF', side=-1),
            'LH': Leg('LH', side=+1),
            'RH': Leg('RH', side=-1),
        }
        self.pos   = np.array([0.0, 0.0])  
        self.theta = 0.0

    def set_q(self, q12):
        q12 = np.asarray(q12)
        for i, name in enumerate(self.LEGS):
            self.legs[name].q = q12[3*i : 3*i+3].copy()

    def get_q(self):
        return np.concatenate([self.legs[n].q for n in self.LEGS])

    def foot_positions(self):
        return {n: self.legs[n].fk() for n in self.LEGS}

    def check_singularities(self, tol=1e-3):
        return {n: self.legs[n].is_singular(tol=tol) for n in self.LEGS}