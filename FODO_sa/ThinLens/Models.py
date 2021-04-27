import math
import json
import typing

import torch
import torch.nn as nn

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps


class TwissFailed(ValueError):
    """Indicate a problem with twiss-calculation."""

    def __init__(self, message):
        self.message = message
        super(TwissFailed, self).__init__(self.message)
        return


class Model(nn.Module):
    def __init__(self, dim: int = 6, slices: int = 1, order: int = 2, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.generalProperties: dict = {"dim": dim, "dtype": dtype, "slices": slices, "order": order}
        self.dim = dim
        self.dtype = dtype

        self.modelType = {"type": type(self).__name__, "dim": dim, "slices": slices, "order": order}

        # log element positions
        self.positions = list()
        self.endPositions = list()

        # must be set by child classes
        self.elements = None
        self.totalLen: float = 0

        return



    def forward(self, x, nTurns: int = 1, outputPerElement: bool = False, outputAtBPM: bool = False):#, twiss: bool = False, index: bool = False):
        # if twiss:
        #     s =  twiss["s"].tolist()
        #     bx = twiss["betx"].tolist()
        #     by = twiss["bety"].tolist()
        #     ax = twiss["alfx"].tolist()
        #     ay = twiss["alfy"].tolist()
        #     if index:
        #         s_new = list()
        #         bx_new = list()
        #         by_new = list()
        #         ax_new = list()
        #         ay_new = list()
        #         for i in index:
        #             s_new.append(s[i])
        #             bx_new.append(s[i])
        #             by_new.append(s[i])
        #             ax_new.append(s[i])
        #             ay_new.append(s[i])
        #         s = s_new
        #         bx = bx_new
        #         by = by_new
        #         ax = ax_new
        #         ay = ay_new
        #         #print('len', len(s))
            
        
        if outputPerElement:
            outputs = list()
            for turn in range(nTurns):
                i = 0
                for e in self.elements:
                    i = i + 1
                    x = e(x)
                    outputs.append(x)
                    
                    #print(self.model.getTunes())
                    # if twiss:
                    #     xPos, yPos = x[:, 0], x[:, 2]
                    #     xp, yp = x[:, 1], x[:, 3]
                    #     #print(i)
                    #     xPosN = xPos / math.sqrt(bx[i])
                    #     xpN = xPos * ax[i] / math.sqrt(bx[i]) + math.sqrt(bx[i]) * xp
                    #     yPosN = yPos / math.sqrt(by[i])
                    #     ypN = yPos * ay[i] / math.sqrt(by[i]) + math.sqrt(by[i]) * yp
                    #     #print( math.sqrt(bx[i]), xPos[0], xPosN[0])
                    #     xT = x.transpose(1, 0)
                    #     xN = torch.stack([xPosN, xpN, yPosN, ypN, *xT[4:]]).transpose(1, 0)
                    #     outputs.append(x)
                    # else:
                    #     outputs.append(x)
            #print(len(outputs), len(twiss["betx"]))

            return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
        elif outputAtBPM:
            #print('dd')
            outputs = list()
            for turn in range(nTurns):
                for e in self.elements:
                    x = e(x)

                    if type(e) is Elements.Monitor:
                        outputs.append(x)

            return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
        else:
            for turn in range(nTurns):
                for e in self.elements:
                    x = e(x)

            return x

    def logElementPositions(self):
        """Store beginning and end of each element."""
        self.positions = list()
        self.endPositions = list()
        totalLength = 0

        for element in self.elements:
            self.positions.append(totalLength)
            totalLength += element.length
            self.endPositions.append(totalLength)

        self.totalLen = totalLength

        return

    def rMatrix(self):
        """Obtain linear transfer matrix."""
        rMatrix = torch.eye(self.dim, dtype=self.dtype)

        for element in self.elements:
            rMatrix = torch.matmul(element.rMatrix(), rMatrix)

        return rMatrix

    def madX(self) -> str:
        """Provide a sequence and necessary templates in order to import this model into Mad-X."""
        # get templates for every map
        elementMaps = list()

        identifier = 0
        for element in self.elements:
            for m in element.maps:
                elementMaps.append(
                    tuple([m.length, m.madX(), identifier]))  # (length, madX element template, identifier)
                identifier += 1

        # create single string containing whole sequence
        templates = ""
        sequence = "sis18: sequence, l = {};\n".format(self.totalLen)

        # add templates and elements
        beginPos = 0  # location of beginning of element (slice) of current map
        for length, template, identifier in elementMaps:
            refPos = beginPos + length / 2  # center of element

            templates += "map{}: ".format(identifier) + template + "\n"
            sequence += "map{}, at={};\n".format(identifier, refPos)

            beginPos += length

        sequence += "\nendsequence;"

        # lattice contains templates and sequence
        lattice = templates + "\n" + sequence
        return lattice

    def thinMultipoleMadX(self):
        """Export as Mad-X sequence consisting of thin-multipole and dipole edge elements."""
        # create single string containing whole sequence
        templates = ""
        sequence = "sis18: sequence, l = {};\n".format(self.totalLen)

        currentPos = 0.0
        kickIdentifier = 0
        for element in self.elements:
            if type(element) is Elements.Drift or type(element) is Elements.Monitor or type(Elements) is Elements.Dummy:
                # drifts are added automatically by Mad-X
                currentPos += element.length
                continue

            for m in element.maps:
                currentPos += m.length

                if type(m) is Maps.DriftMap:
                    # drifts are added automatically by Mad-X
                    continue

                if type(m) is Maps.EdgeKick:
                    # dipole edges cannot be expressed as thin multipoles
                    templates += "dipedge{}: dipedge, ".format(kickIdentifier) + m.thinMultipoleElement() + ";\n"
                    sequence += "dipedge{}, at={};\n".format(kickIdentifier, currentPos)
                else:
                    # add template
                    templates += "kick{}: MULTIPOLE, ".format(kickIdentifier) + m.thinMultipoleElement() + ";\n"
                    sequence += "kick{}, at={};\n".format(kickIdentifier, currentPos)

                kickIdentifier += 1

        sequence += "\nendsequence;"

        # lattice contains templates and sequence
        lattice = templates + "\n" + sequence
        return lattice

    def getTunes(self) -> list:
        """Calculate tune from one-turn map."""
        oneTurnMap = self.rMatrix()

        xTrace = oneTurnMap[:2, :2].trace()
        xTune = torch.acos(1 / 2 * xTrace).item() / (2 * math.pi)

        if self.dim == 4 or self.dim == 6:
            yTrace = oneTurnMap[2:4, 2:4].trace()
            yTune = torch.acos(1 / 2 * yTrace).item() / (2 * math.pi)

            return [xTune, yTune]

        return [xTune, ]

    def getInitialTwiss(self):
        """Calculate twiss parameters of periodic solution at lattice start."""
        oneTurnMap = self.rMatrix()

        # verify absence of coupling
        xyCoupling = oneTurnMap[:2, 2:4]
        yxCoupling = oneTurnMap[2:4, :2]
        couplingIndicator = torch.norm(xyCoupling) + torch.norm(yxCoupling)

        if couplingIndicator != 0:
            raise TwissFailed("coupled motion detected")

        # does a stable solution exist?
        cosMuX = 1 / 2 * oneTurnMap[:2, :2].trace()

        if torch.abs(cosMuX) > 1:
            raise TwissFailed("no periodic solution, cosine(phaseAdvance) out of bounds")

        # calculate twiss from one-turn map
        sinMuX = torch.sign(oneTurnMap[0, 1]) * torch.sqrt(1 - cosMuX ** 2)
        betaX0 = oneTurnMap[0, 1] / sinMuX
        alphaX0 = 1 / (2 * sinMuX) * (oneTurnMap[0, 0] - oneTurnMap[1, 1])

        if self.dim == 4 or self.dim == 6:
            cosMuY = 1 / 2 * oneTurnMap[2:4, 2:4].trace()

            if torch.abs(cosMuX) > 1:
                raise ValueError("no periodic solution, cosine(phaseAdvance) out of bounds")

            sinMuY = torch.sign(oneTurnMap[2, 3]) * torch.sqrt(1 - cosMuY ** 2)
            betaY0 = oneTurnMap[2, 3] / sinMuY
            alphaY0 = 1 / (2 * sinMuY) * (oneTurnMap[2, 2] - oneTurnMap[3, 3])

            return tuple([betaX0, alphaX0]), tuple([betaY0, alphaY0])

        return tuple([betaX0, alphaX0])

    def twissTransportMatrix(self, rMatrix: torch.Tensor):
        """Convert transport matrix into twiss transport matrix."""
        if (self.dim != 4) and (self.dim != 6):
            raise NotImplementedError("twiss calculation not implemented for 2D-case")

        # x-plane
        xMat = rMatrix[:2, :2]
        c, cp, s, sp = xMat[0, 0], xMat[1, 0], xMat[0, 1], xMat[1, 1]

        twissTransportX = torch.tensor([[c ** 2, -2 * s * c, s ** 2],
                                        [-1 * c * cp, s * cp + sp * c, -1 * s * sp],
                                        [cp ** 2, -2 * sp * cp, sp ** 2], ], dtype=self.dtype)

        # x-plane
        yMat = rMatrix[2:4, 2:4]
        c, cp, s, sp = yMat[0, 0], yMat[1, 0], yMat[0, 1], yMat[1, 1]

        twissTransportY = torch.tensor([[c ** 2, -2 * s * c, s ** 2],
                                        [-1 * c * cp, s * cp + sp * c, -1 * s * sp],
                                        [cp ** 2, -2 * sp * cp, sp ** 2], ], dtype=self.dtype)

        return twissTransportX, twissTransportY

    def getTwiss(self):
        if (self.dim != 4) and (self.dim != 6):
            raise NotImplementedError("twiss calculation not implemented for 2D-case")

        # get initial twiss
        twissX0, twissY0 = self.getInitialTwiss()

        pos = [0, ]
        betaX, alphaX, betaY, alphaY = [twissX0[0]], [twissX0[1]], [twissY0[0]], [twissY0[1]]
        twissX0 = torch.tensor([betaX[-1], alphaX[-1], (1 + alphaX[-1] ** 2) / betaX[-1]], dtype=self.dtype)
        twissY0 = torch.tensor([betaY[-1], alphaY[-1], (1 + alphaY[-1] ** 2) / betaY[-1]], dtype=self.dtype)

        lengths = [0, ]
        mux = [0, ]

        # calculate twiss along lattice
        rMatrix = torch.eye(self.dim, dtype=self.dtype)

        for element in self.elements:
            for m in element.maps:
                # update position
                pos.append(pos[-1] + m.length)

                # update twiss
                rMatrix = torch.matmul(m.rMatrix(), rMatrix)
                twissTransportX, twissTransportY = self.twissTransportMatrix(rMatrix)

                twissX = torch.matmul(twissTransportX, twissX0)
                twissY = torch.matmul(twissTransportY, twissY0)

                betaX.append(twissX[0].item())
                alphaX.append(twissX[1].item())
                betaY.append(twissY[0].item())
                alphaY.append(twissY[0].item())

                # # update phase advance
                # lengths.append(m.length)
                # betaXMean = 1/2 * (betaX[-1] + betaX[-2])
                # mux.append(1/betaX[-1] * m.length)

        # store results
        twiss = dict()
        twiss["s"] = torch.tensor(pos, dtype=self.dtype)

        twiss["betx"] = torch.tensor(betaX, dtype=self.dtype)
        twiss["alfx"] = torch.tensor(alphaX, dtype=self.dtype)
        twiss["bety"] = torch.tensor(betaY, dtype=self.dtype)
        twiss["alfy"] = torch.tensor(alphaY, dtype=self.dtype)

        # # calculate phase advance
        # twiss["mux"] = torch.cumsum(torch.tensor(mux, dtype=self.dtype), dim=0)
        # twiss["muy"] = torch.cumsum(1/twiss["bety"], dim=0)
        # # twiss["mux"] = torch.tensor(mux, dtype=self.dtype)
        #
        # lengths = torch.tensor(lengths, dtype=self.dtype)
        # mux = [torch.trapz(1/twiss["betx"][:i+1], lengths[:i+1]) for i in range(len(twiss["betx"]))]
        # twiss["mux"] = torch.tensor(mux, dtype=self.dtype)

        return twiss

    def dumpJSON(self, fileHandler: typing.TextIO):
        """Save model to disk."""
        modelDescription = self.toJSON()

        fileHandler.write(modelDescription)
        return

    def loadJSON(self, fileHandler: typing.TextIO):
        """Load model from disk."""
        modelDescription = fileHandler.read()

        self.fromJSON(modelDescription)
        return

    def toJSON(self):
        """Return model description as string."""
        # store all weights
        weights = [self.modelType, ]
        for e in self.elements:
            weights.append(e.getWeights())

        # return as JSON string
        return json.dumps(weights)

    def fromJSON(self, description: str):
        """Load model from string."""
        weights = iter(json.loads(description))

        # check if model dump is of same type as self
        modelType = next(weights)
        if not modelType == self.modelType:
            print(modelType)
            print(self.modelType)
            raise IOError("file contains wrong model type")

        for e in self.elements:
            e.setWeights(next(weights))

        return


class F0D0Model_single(Model):
    
    def __init__(self, k1f: float = 0.0331, k1d: float = -0.0331, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity
    
        self.cells = list()
        beamline = list()
        for i in range(1):
            # define elements
            dr = Elements.Drift(1.191, **self.generalProperties)
            drb = Elements.Drift(2.618, **self.generalProperties)
            qf = Elements.Quadrupole(4.0, k1f, **self.generalProperties)
            qd = Elements.Quadrupole(4.0, k1d, **self.generalProperties)
            
            cell = [qf, dr, drb, dr,qd, dr, drb, dr]
            beamline.append(cell)
            
        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return
        
class F0D0Model6(Model):
    
    def __init__(self, k1f: float = 0.0331, k1d: float = -0.0331, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity
    
        self.cells = list()
        beamline = list()
        for i in range(6):
            # define elements
            dr = Elements.Drift(1.191, **self.generalProperties)
            drb = Elements.Drift(2.618, **self.generalProperties)
            qf = Elements.Quadrupole(4.0, k1f, **self.generalProperties)
            qd = Elements.Quadrupole(4.0, k1d, **self.generalProperties)
            
            cell = [qf, dr, drb, dr,qd, dr, drb, dr]
            beamline.append(cell)
            
        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return
        
class F0D0Model12(Model):
    
    def __init__(self, k1f: float = 0.0331, k1d: float = -0.0331, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity
    
        self.cells = list()
        beamline = list()
        for i in range(12):
            # define elements
            dr = Elements.Drift(1.191, **self.generalProperties)
            drb = Elements.Drift(2.618, **self.generalProperties)
            qf = Elements.Quadrupole(4.0, k1f, **self.generalProperties)
            qd = Elements.Quadrupole(4.0, k1d, **self.generalProperties)
            
            cell = [qf, dr, drb, dr,qd, dr, drb, dr]
            beamline.append(cell)
            
        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


class RBendLine(Model):
    def __init__(self, angle: float, e1: float, e2: float, dim: int = 4, slices: int = 1, order: int = 2,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # define beam line
        d1 = Elements.Drift(1, **self.generalProperties)
        rb1 = Elements.RBen(0.1, angle, e1=e1, e2=e2, **self.generalProperties)
        d2 = Elements.Drift(1, **self.generalProperties)

        # beam line
        self.elements = nn.ModuleList([d1, rb1, d2])
        self.logElementPositions()
        return


class SIS18_Cell_minimal(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # specify beam line elements
        rb1 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)
        rb2 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)

        d1 = Elements.Drift(0.645, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
        d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
        d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell_minimal_noDipoles(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # specify beam line elements
        d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
        d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, k2f: float = 0, k2d: float = 0,
                 dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # define beam line elements
        rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                             **self.generalProperties)
        rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                             **self.generalProperties)
        rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                             **self.generalProperties)
        rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                             **self.generalProperties)

        # sextupoles
        ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
        ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

        # one day there will be correctors
        hKick1 = Elements.Dummy(0, **self.generalProperties)
        hKick2 = Elements.Dummy(0, **self.generalProperties)
        vKick = Elements.Dummy(0, **self.generalProperties)

        hMon = Elements.Monitor(0.13275, **self.generalProperties)
        vMon = Elements.Monitor(0.13275, **self.generalProperties)

        d1 = Elements.Drift(0.2, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
        d3a = Elements.Drift(6.345, **self.generalProperties)
        d3b = Elements.Drift(0.175, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5a = Elements.Drift(0.195, **self.generalProperties)
        d5b = Elements.Drift(0.195, **self.generalProperties)
        d6a = Elements.Drift(0.3485, **self.generalProperties)
        d6b = Elements.Drift(0.3308, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMon, d6b]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return
        
class SIS18_Lattice_minimal_sa(Model):
    def __init__(self, k1f: float = 0.28339, k1d: float = -0.494471, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        self.cells = list()
        beamline = list()
        for i in range(12):
            # specify beam line elements
            ALPHA = 15 * 1/57.2958
            LL = 150 * 1/57.2958
            PFR = 0.0#7.3 * 1/57.2958
            rb1 = Elements.RBen(length=2.61799, angle=ALPHA, e1=PFR, e2=PFR,
                                **self.generalProperties)
            rb2 = Elements.RBen(length=2.617993878, angle=ALPHA, e1=PFR, e2=PFR,
                                **self.generalProperties)
            # if i == 0 or i == 1 or i == 2:# or i == 3: # or i==1 or i==2: #or i == 5 or i == 9:
            #     qs1f = Elements.QuadrupoleTripplet(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            # else:
            #     qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            # if i == 0 or i == 1 or i == 2: #or i == 3:
            #     #or i==2 :# or i == 1 or i == 2:
            #     qs2d = Elements.QuadrupoleTripplet(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            # else:
            #     qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            # if i == 0 or i == 1 or i == 2: #or i == 5 or i == 9 :
            #     qs3t = Elements.QuadrupoleTripplet(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)
            # else:
            #     qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)
                
            qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            qs3t = Elements.QuadrupoleTripplet(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)
            #qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)
            
            
            d1 = Elements.Drift(0.645, **self.generalProperties)
            d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
            d3 = Elements.Drift(6.8390117, **self.generalProperties)
            d4 = Elements.Drift(0.6000000, **self.generalProperties)
            d5 = Elements.Drift(0.7098000, **self.generalProperties)
            d6 = Elements.Drift(0.4998000, **self.generalProperties)

            cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]
            #cell = [d3, qs1f, d4, qs2d, d5, qs3t, d6, d1, rb1, d2, rb2]
            
            self.cells.append(cell)
            beamline.append(cell)

        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


class SIS18_Lattice_minimal(Model):
    def __init__(self, k1f: float = 2.82632e-01, k1d: float = -4.92e-01, dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        self.cells = list()
        beamline = list()
        for i in range(12):
            # specify beam line elements
            rb1 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                                **self.generalProperties)
            rb2 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                                **self.generalProperties)
            qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)

            d1 = Elements.Drift(0.645, **self.generalProperties)
            d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
            d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
            d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
            d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
            d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

            cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]
            self.cells.append(cell)
            beamline.append(cell)

        # flatten beamlinem
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return
        

#QS1F: QUADRUPOLE, L=1.04, K1=0.282632;
#QS2D: QUADRUPOLE, L=1.04, K1=-0.492;
#QS3T: QUADRUPOLE, L = 0.4804, K1 = 0.656;

class SIS18_Lattice(Model):
    def __init__(self, k1f: float = 2.82632e-01, k1d: float = -4.92e-01, k2f: float = 0, k2d: float = 0,
                 dim: int = 6, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32,
                 cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        self.cells = list()
        beamline = list()
        if cellsIdentical:
            # specify beam line elements
            rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                 **self.generalProperties)
            rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                 **self.generalProperties)
            rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                 **self.generalProperties)
            rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                 **self.generalProperties)

            # sextupoles
            ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
            ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

            # one day there will be correctors
            hKick1 = Elements.Drift(0, **self.generalProperties)
            hKick2 = Elements.Drift(0, **self.generalProperties)
            vKick = Elements.Drift(0, **self.generalProperties)

            hMon = Elements.Monitor(0.13275, **self.generalProperties)
            vMon = Elements.Monitor(0.13275, **self.generalProperties)

            d1 = Elements.Drift(0.2, **self.generalProperties)
            d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
            d3a = Elements.Drift(6.345, **self.generalProperties)
            d3b = Elements.Drift(0.175, **self.generalProperties)
            d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
            d5a = Elements.Drift(0.195, **self.generalProperties)
            d5b = Elements.Drift(0.195, **self.generalProperties)
            d6a = Elements.Drift(0.3485, **self.generalProperties)
            d6b = Elements.Drift(0.3308, **self.generalProperties)

            # quadrupoles shall be sliced more due to their strong influence on tunes
            quadrupoleGeneralProperties = dict(self.generalProperties)
            quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

            qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)

            for i in range(12):
                cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a,
                        ks3c,
                        d5b,
                        qs3t, d6a, hMon, vMon, d6b]

                self.cells.append(cell)
                beamline.append(cell)

        else:
            for i in range(12):
                # specify beam line elements
                rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                     **self.generalProperties)
                rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                     **self.generalProperties)
                rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                     **self.generalProperties)
                rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                     **self.generalProperties)

                # sextupoles
                ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
                ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

                # one day there will be correctors
                hKick1 = Elements.Drift(0, **self.generalProperties)
                hKick2 = Elements.Drift(0, **self.generalProperties)
                vKick = Elements.Drift(0, **self.generalProperties)

                hMon = Elements.Monitor(0.13275, **self.generalProperties)
                vMon = Elements.Monitor(0.13275, **self.generalProperties)

                d1 = Elements.Drift(0.2, **self.generalProperties)
                d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
                d3a = Elements.Drift(6.345, **self.generalProperties)
                d3b = Elements.Drift(0.175, **self.generalProperties)
                d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
                d5a = Elements.Drift(0.195, **self.generalProperties)
                d5b = Elements.Drift(0.195, **self.generalProperties)
                d6a = Elements.Drift(0.3485, **self.generalProperties)
                d6b = Elements.Drift(0.3308, **self.generalProperties)

                # quadrupoles shall be sliced more due to their strong influence on tunes
                quadrupoleGeneralProperties = dict(self.generalProperties)
                quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

                qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
                qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
                qs3t = Elements.Quadrupole(length=0.4804, k1=0.656, **quadrupoleGeneralProperties)

                cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a,
                        ks3c,
                        d5b,
                        qs3t, d6a, hMon, vMon, d6b]

                self.cells.append(cell)
                beamline.append(cell)

        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import ThinLens.Maps

    torch.set_printoptions(precision=4, sci_mode=True)

    dtype = torch.double
    dim = 6

    # set up models
    mod1 = SIS18_Cell(dtype=dtype, dim=dim, slices=4, quadSliceMultiplicity=4)

    # show initial twiss
    print("initial twiss")
    twissX0, twissY0 = mod1.getInitialTwiss()
    print(twissX0, twissY0)

    # get tunes
    print("tunes: {}".format(mod1.getTunes()))

    # show twiss
    twiss = mod1.getTwiss()

    plt.plot(twiss["s"], twiss["betx"])
    plt.show()
    plt.close()

    # dump to string
    modelDescription = mod1.toJSON()
    mod1.fromJSON(modelDescription)

    # dump to file
    with open("/dev/shm/modelDump.json", "w") as f:
        mod1.dumpJSON(f)

    with open("/dev/shm/modelDump.json", "r") as f:
        mod1.loadJSON(f)
