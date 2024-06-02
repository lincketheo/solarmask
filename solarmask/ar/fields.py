from solarmask.utils import *

import numpy as np
import torch
import torch.nn.functional as F

mu0 = 4 * np.pi * 10 ** -3


class Fields:

    def __init__(self, bz, by, bx):
        """A class to write "derived" fields from x, y, z components of magnetic field (and possibly more)
        This class is designed so that derrived fields are only computed once, and 
        operates on an "assertion leve", so that if a field already exists, nothing 
        is done

        Args:
            bz (2d np.array): The z component magnetic flux
            by (2d np.array): The y component magnetic flux
            bx (2d np.array): The x component magnetic flux
        """

        shape_check(bz, by, bx)
        self.shape = bz.shape

        # All the derived fields (private and set to none at first)
        self.__bz = bz
        self.__by = by
        self.__bx = bx

        # Lazily instantiated vars
        self.__hc = None
        self.__bpy = None
        self.__bpx = None
        self.__potential = None
        self.__J = None
        self.__grad_by = None
        self.__grad_by_y = None
        self.__grad_by_x = None
        self.__Jh = None
        self.__grad_bx_y = None
        self.__grad_bx = None
        self.__twist = None
        self.__shear = None
        self.__rho = None
        self.__bpz = None
        self.__grad_bx_x = None
        self.__grad_bz = None
        self.__grad_bz_y = None
        self.__grad_bz_x = None
        self.__grad_bh = None
        self.__grad_bh_y = None
        self.__grad_bh_x = None
        self.__grad_b = None
        self.__grad_b_y = None
        self.__grad_b_x = None
        self.__b = None
        self.__gamma = None
        self.__bh = None

        self.radius = 10
        self.dz = 0.001
        self.__dist_kern = None

    @property
    def bz(self):
        """Line of sight magnetic field

        Returns:
            np.array: A numpy array representing the line of sight magnetic field
        """
        return self.__bz

    @property
    def bx(self):
        """Line of sight magnetic field

        Returns:
            np.array: A numpy array representing the line of sight magnetic field
        """
        return self.__bx

    @property
    def by(self):
        """Line of sight magnetic field

        Returns:
            np.array: A numpy array representing the line of sight magnetic field
        """
        return self.__by

    @property
    def bh(self):
        """Horizontal Magnetic Field Component

        $$b_h = norm(b_x, b_y)$$
        """
        if self.__bh is None:
            self.__bh = p_norm((self.bx, self.by))
        return self.__bh

    @property
    def gamma(self):
        """Angle of line of magnetic flux vector from the horizontal

        $$\\gamma = arctan(\\frac{b_z}{|b_x| + |b_y|})$$
        """
        if self.__gamma is None:
            self.__gamma = np.arctan(self.bz / self.bh)
        return self.__gamma

    @property
    def b(self):
        """Magnitude of magnetic flux vector

        $$b = norm(b_x, b_y, b_z)$$
        """
        if self.__b is None:
            self.__b = p_norm((self.bx, self.by, self.bz))
        return self.__b

    @property
    def grad_b_x(self):
        """Gradient of magnetic field magnitude

        $$b = norm(b_x, b_y, b_z)$$
        """
        if self.__grad_b_x is None or self.__grad_b_y is None:
            self.__grad_b_x, self.__grad_b_y = gradient(self.b)
        return self.__grad_b_x

    @property
    def grad_b_y(self):
        """Gradient of magnetic field magnitude

        $$b = norm(b_x, b_y, b_z)$$
        """
        if self.__grad_b_x is None or self.__grad_b_y is None:
            self.__grad_b_x, self.__grad_b_y = gradient(self.b)
        return self.__grad_b_y

    @property
    def grad_b(self):
        if self.__grad_b is None:
            self.__grad_b = p_norm((self.grad_b_x, self.grad_b_y))
        return self.__grad_b

    @property
    def grad_bh_x(self):
        """Gradient of horizontal magnetic field

        $$b_h = norm(b_x, b_y)$$
        """
        if self.__grad_bh_x is None or self.__grad_bh_y is None:
            self.__grad_bh_x, self.__grad_bh_y = gradient(self.bh)
        return self.__grad_bh_x

    @property
    def grad_bh_y(self):
        """Gradient of horizontal magnetic field

        $$b_h = norm(b_x, b_y)$$
        """
        if self.__grad_bh_x is None or self.__grad_bh_y is None:
            self.__grad_bh_x, self.__grad_bh_y = gradient(self.bh)
        return self.__grad_bh_y

    @property
    def grad_bh(self):
        if self.__grad_bh is None:
            self.__grad_bh = p_norm((self.grad_bh_x, self.grad_bh_y))
        return self.__grad_bh

    @property
    def grad_bz_x(self):
        """Gradient of line of site magnetic field
        """
        if self.__grad_bz_x is None or self.__grad_bz_y is None:
            self.__grad_bz_x, self.__grad_bz_y = gradient(self.bz)
        return self.__grad_bz_x

    @property
    def grad_bz_y(self):
        """Gradient of line of site magnetic field
        """
        if self.__grad_bz_x is None or self.__grad_bz_y is None:
            self.__grad_bz_x, self.__grad_bz_y = gradient(self.bz)
        return self.__grad_bz_y

    @property
    def grad_bz(self):
        if self.__grad_bz is None:
            self.__grad_bz = p_norm((self.grad_bz_x, self.grad_bz_y))
        return self.__grad_bz

    @property
    def grad_bx_x(self):
        """Gradient of line of x component of magnetic field
        """
        if self.__grad_bx_x is None or self.__grad_bx_y is None:
            self.__grad_bx_x, self.__grad_bx_y = gradient(self.bx)
        return self.__grad_bx_x

    @property
    def grad_bx_y(self):
        """Gradient of line of x component of magnetic field
        """
        if self.__grad_bx_x is None or self.__grad_bx_y is None:
            self.__grad_bx_x, self.__grad_bx_y = gradient(self.bx)
        return self.__grad_bx_y

    @property
    def grad_bx(self):
        if self.__grad_bx is None:
            self.__grad_bx = p_norm((self.grad_bx_x, self.grad_bx_y))
        return self.__grad_bx

    @property
    def grad_by_x(self):
        """Gradient of line of x component of magnetic field
        """
        if self.__grad_by_x is None or self.__grad_by_y is None:
            self.__grad_by_x, self.__grad_by_y = gradient(self.bx)
        return self.__grad_by_x

    @property
    def grad_by_y(self):
        """Gradient of line of x component of magnetic field
        """
        if self.__grad_by_x is None or self.__grad_by_y is None:
            self.__grad_by_x, self.__grad_by_y = gradient(self.bx)
        return self.__grad_by_y

    @property
    def grad_by(self):
        if self.__grad_by is None:
            self.__grad_by = p_norm((self.grad_by_x, self.grad_by_y))
        return self.__grad_by

    @property
    def j(self):
        """Vertical current density

        $$J_z = \\frac{\\partial b_y}{\\partial x} - \\frac{\\partial b_x}{\\partial y}$$
        """
        if self.__J is None:
            self.__J = (self.grad_by_x - self.grad_bx_y) / mu0
            self.__J[self.bz == 0] = 0
        return self.__J

    @property
    def jh(self):
        """Vertical heterogeneity current density

        $$J_z^h = \\frac{1}{b}(b_y\\frac{\\partial b_x}{\\partial y} - b_x \\frac{\\partial b_y}{\\partial x})$$
        """
        if self.__Jh is None:
            self.__Jh = (self.by * self.grad_bx_y - self.bx * self.grad_by_x) / mu0
        return self.__Jh

    @property
    def hc(self):
        """Current helicity

        $$h_c = b_z(\\frac{\\partial b_y}{\\partial x} - \\frac{\\partial b_x}{\\partial y})$$
        """
        if self.__hc is None:
            self.__hc = self.bz * self.j
        return self.__hc

    @property
    def twist(self):
        """Twist
        """
        if self.__twist is None:
            self.__twist = np.divide(self.j, self.bz, out=np.ones_like(self.j), where=self.bz != 0)
        return self.__twist

    @property
    def shear(self):
        """Shear angle

        $$\\Psi = arccos(\\frac{\\vec{b_p}\\cdot\\vec{b_o}}{|b_o||b_p|})$$
        """
        if self.__shear is None:
            dot = self.bx * self.bpx + self.by * self.bpy + self.bz * self.bpz
            magp = p_norm((self.bpx, self.bpy, self.bpz))
            self.__shear = np.arccos(dot / (self.b * magp))
        return self.__shear

    @property
    def rho(self):
        """Excess magnetic energy density

        $$\\rho_e = norm(b_p - b_o)$$
        """
        if self.__rho is None:
            self.__rho = (self.b - p_norm((self.bpx, self.bpy, self.bpz))) ** 2 / (8 * np.pi)
        return self.__rho

    @property
    def bpx(self):
        if self.__bpx is None:
            self.greenpot()
        return self.__bpx

    @property
    def bpy(self):
        if self.__bpy is None:
            self.greenpot()
        return self.__bpy

    @property
    def bpz(self):
        if self.__bpz is None:
            self.__bpz = self.__bz
        return self.__bpz

    def greenpot(self):
        """Magnetic field vector assuming zero current

        $$b = -\\nabla \\phi \\quad \\nabla^2 \\phi= 0 \\quad (z > 0)$$
        $$-\\hat{n}\\cdot\\nabla\\phi = b_n \\quad (z = 0)$$
        $$\\phi(r) \\rightarrow 0 \\quad as \\quad r \\rightarrow \\infty \\quad (z = 0)$$

        Use green's function for the neumann boundary condition
        $$\\nabla^2 G_n(r, r') = 0 \\quad (z > 0)$$
        $$G_n \\rightarrow 0 \\quad as \\quad |r - r'| \\rightarrow \\infty \\quad (z > 0)$$
        $$-\\hat{n}\\cdot\\nabla G_n = 0 \\quad (z = 0, r' \\neq r)$$

        $$-\\hat{n}\\cdot\\nabla G_n$$ 
        
        diverges to keep unit flux

        $$lim_{z \\rightarrow 0^+}\\int\\hat{n}\\cdot\\nabla G_n(r,r')dS = 1$$
        
        We have the solution:

        $$\\phi(r) = \\int b_n(r')G_n(r, r')dS' \\quad dS' = dx'dy' \\quad r' = (x', y', 0)$$

        Explicit form:
        $$G_n(r, r') = \\frac{1}{2\\phi R} \\quad (R = |r - r'|)$$
        
        Discrete form:
        $$\\phi(r) \\rightarrow \\sum_{r'_{ij}}b_n(r')\\tilde{G}_n(r, r'_{ij})\\Delta^2$$
        
        and $$b_n$$ is approximated by:

        $$b_n(r) \\rightarrow -\\sum_{r'_{ij}}\\hat{n}\\cdot\\nabla \\tilde{G}_n(r, r'_{ij})\\Delta^2 \\quad (r = (x, y, 0))$$
        
        $$\\tilde{G}_n(r, r'_{ij}) = \\frac{1}{2\\pi |r - r'_{ij} + (\\Delta/\\sqrt{2\\pi}\\hat{n})|}$$

        As if there is a magnetic pole located just 
        
        $$\\Delta/\\sqrt{2\\pi}$$ 
        
        below the surface
        """
        # Transfer to PyTorch
        radius = self.radius
        gpu_dev = get_gpu_dev()
        bz = F.pad(torch.from_numpy(self.bz), (radius, radius, radius, radius)).float()
        bz = bz.to(gpu_dev)

        # Distance kernel - a kernel with values filled in the "circle" (by def of norm) as the distance from
        # the center multiplied by dz (for integration)

        Gn = self._dist_kern.to(gpu_dev)

        # Convolution -- integrate over each pixel
        pot = F.conv2d(bz[None, None, ...], Gn[None, None, ...])[0][0].cpu().numpy()

        # Save potential
        self.__potential = pot

        # Get Potential Fields
        grad = gradient(self.__potential)
        self.__bpx, self.__bpy = -grad[0], -grad[1]

    @property
    def _dist_kern(self):
        if self.__dist_kern is None:
            # The kernel to use greens function (radius 10)
            radius = self.radius
            dz = self.dz

            dist_kern = torch.zeros((2 * radius + 1, 2 * radius + 1))
            for x0 in range(radius + 1):
                for y0 in range(radius + 1):
                    if p_norm((x0, y0)) <= radius:
                        v = dz / p_norm((x0, y0, dz))
                        dist_kern[radius + x0][radius + y0] = v
                        dist_kern[radius - x0][radius + y0] = v
                        dist_kern[radius + x0][radius - y0] = v
                        dist_kern[radius - x0][radius - y0] = v
            self.__dist_kern = dist_kern
        return self.__dist_kern
