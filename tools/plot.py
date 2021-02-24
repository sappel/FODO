import torch
import matplotlib.pyplot as plt


def track(model, bunch, turns: int):
    device = next(model.parameters()).device
    bunch.to(device)

    # track
    with torch.no_grad():

        multiTurnOutput = list()
        y = bunch
        for i in range(turns):
            y = model(y, outputPerElement=True)
            multiTurnOutput.append(y)
            y = y[:, :, -1]

    # prepare tracks for plotting
    trackResults = torch.cat(multiTurnOutput, 2)  # indices: particle, dim, element
    return trackResults


def trajectories(ax: plt.axes, trackResults, lattice, plane: str = "x"):
    """Plot individual trajectories."""
    trackResults = trackResults.to("cpu")

    pos = [lattice.endPositions[i % len(lattice.endPositions)] + i // len(lattice.endPositions) * lattice.totalLen
           for i in range(trackResults.size(2))]

    for particle in trackResults:
        if plane == "x":
            ax.plot(pos, particle[0])
        elif plane == "y":
            ax.plot(pos, particle[2])
        else:
            raise ValueError("invalid choice of plane")

    return
