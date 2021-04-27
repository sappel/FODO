import math

import torch
import torch.distributions


class Beam(object):
    def __init__(self, mass: float, energy: float, exn: float, eyn: float, sigt: float, sige: float, particles: int,
                 charge: int = 1, centroid: list = (0, 0, 0, 0, 0)):
        """
        Set up beam including bunch of individual particles.
        """
        # calculate properties of reference particle
        self.energy = energy  # GeV
        self.mass = mass  # GeV
        self.charge = charge  # e
        self.exn = exn
        self.eyn = eyn

        self.gamma = self.energy / self.mass
        self.momentum = torch.sqrt(torch.tensor(self.energy) ** 2 - torch.tensor(self.mass) ** 2).item()  # GeV/c

        self.beta = self.momentum / (self.gamma * self.mass)
        
        # standard deviations assuming round beams
        ex = exn / (self.beta * self.gamma)  # m
        ey = eyn / (self.beta * self.gamma)  # m

        stdX = math.sqrt(ex / math.pi)
        stdY = math.sqrt(ey / math.pi)

        stdE = sige * self.energy  # GeV

        std = torch.FloatTensor([stdX, stdX, stdY, stdY, sigt, stdE])

        # sample particles
        loc = torch.FloatTensor([*centroid, self.energy])
        scaleTril = torch.diag(std ** 2)
        #print(scaleTril)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=scaleTril)

        preliminaryBunch = dist.sample((particles,))  # x, xp, y, yp, sigma, totalEnergy

        # calculate missing properties for individual particles
        pSigma = (preliminaryBunch[:, 5] - self.energy) / (self.beta * self.momentum)

        momentum = torch.sqrt(preliminaryBunch[:, 5] ** 2 - self.mass ** 2)

        delta = (momentum - self.momentum) / self.momentum
        invDelta = 1 / (delta + 1)
        gamma = preliminaryBunch[:, 5] / self.mass
        beta = momentum / (gamma * self.mass)
        velocityRatio = self.beta / beta

        # create bunch
        x = preliminaryBunch[:, 0]
        xp = preliminaryBunch[:, 1]
        y = preliminaryBunch[:, 2]
        yp = preliminaryBunch[:, 3]
        sigma = preliminaryBunch[:, 4]
        energy = preliminaryBunch[:, 5]

        self.bunch = torch.stack([x, xp, y, yp, sigma, pSigma, delta, invDelta, velocityRatio, ]).t()

        # check if nan occurs in bunch <- can be if sige is too large and hence energy is smaller than rest energy
        assert not self.bunch.isnan().any()

        return

    def fromDelta(self, delta: torch.tensor):
        """Set particle momentum deviation to delta and adjust coordinates accordingly."""
        # calculate properties
        invDelta = 1 / (delta + 1)

        momentum = self.momentum * delta + self.momentum
        energy = torch.sqrt(momentum ** 2 + self.mass ** 2)
        gamma = energy / self.mass
        beta = momentum / (gamma * self.mass)

        pSigma = (energy - self.energy) / (self.beta * self.momentum)
        velocityRatio = self.beta / beta

        # select and update particles
        bunch = self.bunch[:len(delta)].t()
        bunch = torch.stack([*bunch[:5], pSigma, delta, invDelta, velocityRatio])
        return bunch.t()

    def madX(self):
        """Export as arguments for madx.beam command."""
        return {"mass": self.mass, "charge": self.charge, "exn": self.exn, "eyn": self.eyn, "gamma": self.gamma}


if __name__ == "__main__":
    torch.set_printoptions(precision=2, sci_mode=True)

    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.005, particles=int(1e1))

    print(beam.bunch)
