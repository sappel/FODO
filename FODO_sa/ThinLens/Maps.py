import math
import torch
import torch.nn as nn


class Map(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.length = 0.0
        return

    def madX(self) -> str:
        """Express this map via "arbitrary matrix" element from MAD-X."""
        rMatrix = self.rMatrix()

        elementDefinition = "MATRIX, L={}".format(self.length)

        for i in range(len(rMatrix)):
            for j in range(len(rMatrix[0])):
                elementDefinition += ", RM{}{}={}".format(i + 1, j + 1, rMatrix[i, j])

        elementDefinition += ";"
        return elementDefinition


class DriftMap(Map):
    """Propagate bunch along drift section."""

    def __init__(self, length: float, dim: int, dtype: torch.dtype):
        super().__init__(dim, dtype)
        self.length = length

        # set up weights
        kernel = torch.tensor(self.length, dtype=self.dtype)
        self.weight = nn.Parameter(kernel)

        if dim == 4:
            self.forward = self.forward4D
        elif dim == 6:
            self.forward = self.forward6D
        else:
            raise NotImplementedError("dim {} not supported".format(dim))

        return

    def forward4D(self, x):
        # get momenta
        momenta = x[:, [1, 3]]

        # get updated positions
        pos = self.length * momenta
        pos = pos + x[:, [0, 2]]

        # update phase space vector
        xT = x.transpose(1, 0)
        posT = pos.transpose(1, 0)

        x = torch.stack([posT[0], xT[1], posT[1], xT[3]], ).transpose(1, 0)

        return x

    def forward6D(self, x):
        # get momenta
        momenta = x[:, [1, 3, ]]
        velocityRatio = x[:, 8]

        # get updated momenta
        pos = self.weight * momenta
        sigma = self.weight * (1 - velocityRatio)
        pos = pos + x[:, [0, 2]]
        sigma = sigma + x[:, 4]

        # update phase space vector
        xT = x.transpose(1, 0)
        posT = pos.transpose(1, 0)

        x = torch.stack([posT[0], xT[1], posT[1], xT[3], sigma, *xT[5:]], ).transpose(1, 0)
        return x

    def rMatrix(self):
        if self.dim == 4:
            rMatrix = torch.eye(4, dtype=self.dtype)
            rMatrix[0, 1] = self.weight
            rMatrix[2, 3] = self.weight
        else:
            rMatrix = torch.eye(6, dtype=self.dtype)
            rMatrix[0, 1] = self.weight
            rMatrix[2, 3] = self.weight

        return rMatrix


class DipoleKick(Map):
    """Apply an horizontal dipole kick."""

    def __init__(self, length: float, angle: float, dim: int, dtype: torch.dtype):
        super().__init__(dim=dim, dtype=dtype)
        self.dipoleLength = length  # used to calculate k0L for Mad-X

        # initialize weight
        curvature = angle / length
        kernel = torch.tensor(curvature, dtype=dtype)
        self.weight = nn.Parameter(kernel)

        if dim == 4:
            self.forward = self.forward4D
        elif dim == 6:
            self.forward = self.forward6D
        else:
            raise NotImplementedError("dim {} not supported".format(dim))

        return

    def forward4D(self, x):
        # get horizontal position
        pos = x[:, 0]

        # get updated momenta
        momenta = -1 * self.weight ** 2 * self.dipoleLength * pos
        momenta = momenta + x[:, 1]

        # update phase space vector
        xT = x.transpose(1, 0)

        x = torch.stack([xT[0], momenta, xT[2], xT[3]], ).transpose(1, 0)
        return x

    def forward6D(self, x):
        # get x and sigma
        pos = x[:, [0, 4]]

        delta = x[:, 6]
        velocityRatio = x[:, 8]

        # get updates
        px = -1 * self.weight ** 2 * self.dipoleLength * pos[:, 0] + self.weight * self.dipoleLength * delta
        px = x[:, 1] + px

        sigma = -1 * self.weight * self.dipoleLength * velocityRatio
        sigma = x[:, 4] + sigma

        # update phase space vector
        xT = x.transpose(1, 0)

        x = torch.stack([xT[0], px, xT[2], xT[3], sigma, *xT[5:]]).transpose(1, 0)
        return x

    def rMatrix(self):
        if self.dim == 4:
            rMatrix = torch.eye(4, dtype=self.dtype)
            rMatrix[1, 0] = -1 * self.weight ** 2 * self.dipoleLength
        else:
            rMatrix = torch.eye(6, dtype=self.dtype)
            rMatrix[1, 0] = -1 * self.weight ** 2 * self.dipoleLength
            rMatrix[4, 0] = -1 * self.weight * self.dipoleLength

        return rMatrix

    def thinMultipoleElement(self):
        k0L = self.weight.item() * self.dipoleLength  # horizontal bending angle
        return "LRAD={length}, KNL={{{k0L}}}".format(length=self.dipoleLength, k0L=k0L)


class EdgeKick(Map):
    """Dipole edge effects."""

    def __init__(self, length: float, bendAngle: float, edgeAngle: float, dim: int, dtype: torch.dtype):
        super().__init__(dim=dim, dtype=dtype)

        self.curvature = bendAngle / length
        self.edgeAngle = edgeAngle

        # initialize weight
        kernel = torch.tensor(math.tan(self.edgeAngle), dtype=dtype)
        self.weight = nn.Parameter(kernel)

        if dim == 4:
            self.forward = self.forward4D
        elif dim == 6:
            self.forward = self.forward6D
        else:
            raise NotImplementedError("dim {} not supported".format(dim))

        return

    def forward4D(self, x):
        # get positions
        pos = x[:, [0, 2]]

        # get updated momenta
        momenta = self.curvature * self.weight * torch.tensor([1, -1], dtype=self.dtype) * pos
        momenta = momenta + x[:, [1, 3]]

        # update phase space vector
        xT = x.transpose(1, 0)
        momentaT = momenta.transpose(1, 0)

        x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1]], ).transpose(1, 0)
        return x

    def forward6D(self, x):
        # get positions
        pos = x[:, [0, 2]]
        velocityRatio = x[:, 8]

        # get updated momenta
        momenta = self.curvature * self.weight * torch.tensor([1, -1], dtype=self.dtype) * pos
        momenta = momenta + x[:, [1, 3]]

        # sigma = -1 * velocityRatio * pos[0] * self.weight[2]
        # sigma = sigma + x[:, 4]

        # update phase space vector
        xT = x.transpose(1, 0)
        momentaT = momenta.transpose(1, 0)

        x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1], *xT[4:], ]).transpose(1, 0)
        return x

    def rMatrix(self):
        if self.dim == 4:
            rMatrix = torch.eye(4, dtype=self.dtype)
        else:
            rMatrix = torch.eye(6, dtype=self.dtype)

        rMatrix[1, 0] = self.curvature * math.tan(self.edgeAngle)
        rMatrix[3, 2] = -1 * self.curvature * math.tan(self.edgeAngle)

        return rMatrix

    def thinMultipoleElement(self):
        return "H={}, E1={}".format(self.curvature, self.edgeAngle)


class MultipoleKick(Map):
    def __init__(self, length: float, dim: int, dtype: torch.dtype, kn: list = None, ks: list = None):
        super().__init__(dim, dtype)
        self.length = 0.0  # dummy length used to locate kick / drift locations along the ring
        self.kickLength = length  # length used to calculate integrated multipole strengths

        # register multipole strengths as weights
        if kn:
            for i in range(1, 4):
                weight = torch.tensor([kn[i - 1]], dtype=dtype)
                self.register_parameter("k{}n".format(i), nn.Parameter(weight))
        else:
            self.k1n = nn.Parameter(torch.tensor([0, ], dtype=dtype))
            self.k2n = nn.Parameter(torch.tensor([0, ], dtype=dtype))
            self.k3n = nn.Parameter(torch.tensor([0, ], dtype=dtype))

        if ks:
            for i in range(1, 4):
                weight = torch.tensor([ks[i - 1]], dtype=dtype)
                self.register_parameter("k{}s".format(i), nn.Parameter(weight))
        else:
            self.k1s = nn.Parameter(torch.tensor([0, ], dtype=dtype))
            self.k2s = nn.Parameter(torch.tensor([0, ], dtype=dtype))
            self.k3s = nn.Parameter(torch.tensor([0, ], dtype=dtype))

        if dim == 6:
            self.forward = self.forward6D
        else:
            raise NotImplementedError("dim {} not supported".format(dim))

        return

    def forward6D(self, x):
        # get positions
        xPos, yPos = x[:, 0], x[:, 2]
        invDelta = x[:, 7]

        # get updated momenta
        quadDPx = self.k1n * xPos - self.k1s * yPos
        quadDPy = self.k1n * yPos + self.k1s * xPos
        
        sextDPx = self.k2n * 1 / 2 * (xPos ** 2 + yPos ** 2) - self.k2s * xPos * yPos
        sextDPy = self.k2n * xPos * yPos + self.k2s * 1 / 2 * (xPos ** 2 + yPos ** 2)

        octDPx = self.k3n * (1 / 6 * xPos ** 3 - 1 / 2 * xPos * yPos ** 2) + self.k3s * (
                    1 / 6 * yPos ** 3 - 1 / 2 * xPos ** 2 * yPos)
        octDPy = self.k3n * (-1 / 6 * yPos ** 3 + 1 / 2 * xPos ** 2 * yPos) + self.k3s * (
                    1 / 6 * xPos ** 3 - 1 / 2 * xPos * yPos ** 2)

        px = x[:, 1] - self.kickLength * invDelta * (quadDPx + sextDPx + octDPx)
        py = x[:, 3] + self.kickLength * invDelta * (quadDPy + sextDPy + octDPy)

        # update phase space vector
        xT = x.transpose(1, 0)
        x = torch.stack([xT[0], px, xT[2], py, *xT[4:]]).transpose(1, 0)
        return x

    def rMatrix(self):
        """Calculate transfer matrix considering only linear optics."""
        if self.dim == 4:
            rMatrix = torch.eye(4, dtype=self.dtype)
        else:
            rMatrix = torch.eye(6, dtype=self.dtype)

        rMatrix[1, 0] = -1 * self.kickLength * self.k1n
        rMatrix[3, 2] = self.kickLength * self.k1n

        rMatrix[1, 2] = self.kickLength * self.k1s
        rMatrix[3, 0] = self.kickLength * self.k1s

        return rMatrix

    def thinMultipoleElement(self):
        integratedMultipoleStrengths = dict()
        integratedMultipoleStrengths["k1nl"] = self.kickLength * self.k1n.item()
        integratedMultipoleStrengths["k2nl"] = self.kickLength * self.k2n.item()
        integratedMultipoleStrengths["k3nl"] = self.kickLength * self.k3n.item()

        integratedMultipoleStrengths["k1sl"] = self.kickLength * self.k1s.item()
        integratedMultipoleStrengths["k2sl"] = self.kickLength * self.k2s.item()
        integratedMultipoleStrengths["k3sl"] = self.kickLength * self.k3s.item()

        return "KNL={{0.0, {k1nl}, {k2nl}, {k3nl}}}, KSL={{0.0, {k1sl}, {k2sl}, {k3sl}}}".format(
            **integratedMultipoleStrengths)


if __name__ == "__main__":
    dim = 6
    dtype = torch.double

    # set up quad
    quad = MultipoleKick(1, dim, dtype, ks=[1, 0.3, 0.1], )

    # track
    x = torch.randn((2, 9), dtype=dtype)
    print(x)
    print(quad(x))

    # matrix
    print("rMatrix")
    print(quad.rMatrix())
    print(quad.thinMultipoleElement())
