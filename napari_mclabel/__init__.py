try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from .mclabel import napari_experimental_provide_dock_widget

__all__ = ["napari_experimental_provide_dock_widget"]