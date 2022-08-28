import polyscope_bindings as psb
import os
import numpy as np

from polyscope.core import str_to_datatype, glm3

def remove_floating_quantity(name, error_if_absent=False):
    psb.remove_floating_quantity(name, error_if_absent)

def remove_all_floating_quantities():
    psb.remove_all_floating_quantities()

def add_scalar_image_quantity(name, values, enabled=None, datatype="standard", vminmax=None, cmap=None, show_fullscreen=None):

    # Check image data
    shape_msg = "`values` should be a 2d [width,height] numpy array"
    if len(values.shape) != 2:
        raise ValueError(shape_msg)
    w = values.shape[0]
    h = values.shape[1]
    values = values.flatten() # flatten it

    # Add the actual quantity
    q = psb.add_scalar_image_quantity(name, w, h, values, str_to_datatype(datatype))

    # Support optional params
    if enabled is not None:
        q.set_enabled(enabled)
    if vminmax is not None:
        q.set_map_range(vminmax)
    if cmap is not None:
        q.set_color_map(cmap)
    if show_fullscreen is not None:
        q.set_show_fullscreen(show_fullscreen)

def add_color_image_quantity(name, color_values, enabled=None, show_fullscreen=None):

    # Check image data
    shape_msg = "`color_values` should be a 3d [width,height,3] numpy array"
    if len(color_values.shape) != 3 or color_values.shape[2] != 3:
        raise ValueError(shape_msg)

    # flatten it
    w = color_values.shape[0]
    h = color_values.shape[1]
    color_values = color_values.reshape((w*h, 3))

    # Add the actual quantity
    q = psb.add_color_image_quantity(name, w, h, color_values)

    # Support optional params
    if enabled is not None:
        q.set_enabled(enabled)
    if show_fullscreen is not None:
        q.set_show_fullscreen(show_fullscreen)


def add_depth_render_image_quantity(name, depth_values, normal_values, enabled=None, color=None, material=None, transparency=None):

    # Check image data
    if len(depth_values.shape) != 2:
        raise ValueError("`depth_values` should be a 2d [width,height] numpy array.")
    if len(normal_values.shape) != 3 or normal_values.shape[2] != 3:
        raise ValueError("`normal_values` should be a 3d [width,height,3] numpy array.")
    if depth_values.shape[:2] != normal_values.shape[:2]:
        raise ValueError("all input value arrays should have the same size in the first 2 dimensions")
       
     # flatten it
    w = depth_values.shape[0]
    h = depth_values.shape[1]
    depth_values = depth_values.flatten()
    normal_values = normal_values.reshape(-1,3)

    # Add the actual quantity
    q = psb.add_depth_render_image_quantity(name, w, h, depth_values, normal_values)

    # Support optional params
    if enabled is not None:
        q.set_enabled(enabled)
    if color is not None:
        q.set_color(glm3(color))
    if material is not None:
        q.set_material(material)
    if transparency is not None:
        q.set_transparency(transparency)


def add_color_render_image_quantity(name, depth_values, normal_values, color_values, enabled=None, material=None, transparency=None):

    # Check image data
    if len(depth_values.shape) != 2:
        raise ValueError("`depth_values` should be a 2d [width,height] numpy array.")
    if len(normal_values.shape) != 3 or normal_values.shape[2] != 3:
        raise ValueError("`normal_values` should be a 3d [width,height,3] numpy array.")
    if len(color_values.shape) != 3 or color_values.shape[2] != 3:
        raise ValueError("`color_values` should be a 3d [width,height,3] numpy array.")
    if depth_values.shape[:2] != normal_values.shape[:2] or \
       depth_values.shape[:2] != color_values.shape[:2]:
        raise ValueError("all input value arrays should have the same size in the first 2 dimensions")
      
    # flatten it
    w = depth_values.shape[0]
    h = depth_values.shape[1]
    depth_values = depth_values.flatten() 
    normal_values = normal_values.reshape(-1,3)
    color_values = color_values.reshape(-1,3)

    # Add the actual quantity
    q = psb.add_color_render_image_quantity(name, w, h, depth_values, normal_values, color_values)

    # Support optional params
    if enabled is not None:
        q.set_enabled(enabled)
    if material is not None:
        q.set_material(material)
    if transparency is not None:
        q.set_transparency(transparency)


def add_scalar_render_image_quantity(name, depth_values, normal_values, scalar_values, enabled=None, material=None, transparency=None, datatype="standard", vminmax=None, cmap=None):

    # Check image data
    if len(depth_values.shape) != 2:
        raise ValueError("`depth_values` should be a 2d [width,height] numpy array.")
    if len(normal_values.shape) != 3 or normal_values.shape[2] != 3:
        raise ValueError("`normal_values` should be a 3d [width,height,3] numpy array.")
    if len(scalar_values.shape) != 2:
        raise ValueError("`scalar_values` should be a 2d [width,height] numpy array.")
    if depth_values.shape[:2] != normal_values.shape[:2] or \
       depth_values.shape[:2] != scalar_values.shape[:2]:
        raise ValueError("all input value arrays should have the same size in the first 2 dimensions")

    # flatten it
    w = depth_values.shape[0]
    h = depth_values.shape[1]
    depth_values = depth_values.flatten() 
    normal_values = normal_values.reshape(-1,3) 
    scalar_values = scalar_values.flatten()

    # Add the actual quantity
    q = psb.add_scalar_render_image_quantity(name, w, h, depth_values, normal_values, scalar_values, str_to_datatype(datatype))

    # Support optional params
    if enabled is not None:
        q.set_enabled(enabled)
    if material is not None:
        q.set_material(material)
    if transparency is not None:
        q.set_transparency(transparency)
    if vminmax is not None:
        q.set_map_range(vminmax)
    if cmap is not None:
        q.set_color_map(cmap)
