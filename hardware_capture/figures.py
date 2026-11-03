"""Create file-only figures without initializing a GUI toolkit."""


def subplots(*args, show=False, figsize=None, **kwargs):
    if show:
        import matplotlib.pyplot as plt
        return plt.subplots(*args, figsize=figsize, **kwargs)
    from matplotlib.figure import Figure
    figure = Figure(figsize=figsize)
    return figure, figure.subplots(*args, **kwargs)
