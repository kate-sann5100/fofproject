import os, sys
import plotly, plotly.io as pio
print("Python:", sys.version)
print("Plotly:", plotly.__version__)
print("Renderer (pio):", pio.renderers.default)
print("PLOTLY_RENDERER env:", os.environ.get("PLOTLY_RENDERER"))
print("Available renderers:", pio.renderers.names)