import polyscope_bindings as psb

from polyscope.core import str_to_datatype, str_to_vectortype, glm3
   
def implicit_render_opts_object_from_args(mode=None, step_factor=None, normal_sample_eps=None, n_max_steps=None, subsample_factor=None, miss_dist=None, autoscale_miss_dist=True, hit_dist=None, autoscale_hit_dist=True, step_size=None, autoscale_step_size=True):
    opts = psb.ImplicitRenderOpts()
    if mode is not None:
        opts.mode = mode
    if step_factor is not None:
        opts.step_factor = step_factor
    if normal_sample_eps is not None:
        opts.normal_sample_eps = normal_sample_eps
    if n_max_steps is not None:
        opts.n_max_steps = n_max_steps
    if subsample_factor is not None:
        opts.subsample_factor = subsample_factor
    if miss_dist is not None:
        opts.set_miss_dist(miss_dist, autoscale_miss_dist)
    if hit_dist is not None:
        opts.set_hit_dist(hit_dist, autoscale_hit_dist)
    if step_size is not None:
        opts.set_step_size(step_size, autoscale_step_size)
    return opts

def render_implicit_surface(name, func, enabled=None, material=None, transparency=None, color=None, **render_opts_kwargs):

    opts = implicit_render_opts_object_from_args(**render_opts_kwargs)

    q = psb.render_implicit_surface(name, func, opts)

    # Support optional params
    if enabled is not None:
        q.set_enabled(enabled)
    if material is not None:
        q.set_material(material)
    if transparency is not None:
        q.set_transparency(transparency)
    if color is not None:
        q.set_color(glm3(color))

def render_implicit_surface_color(name, func, func_color, enabled=None, material=None, transparency=None, **render_opts_kwargs):

    opts = implicit_render_opts_object_from_args(**render_opts_kwargs)

    q = psb.render_implicit_surface_color(name, func, func_color, opts)

    # Support optional params
    if enabled is not None:
        q.set_enabled(enabled)
    if material is not None:
        q.set_material(material)
    if transparency is not None:
        q.set_transparency(transparency)

def render_implicit_surface_scalar(name, func, func_scalar, enabled=None, material=None, transparency=None, datatype="standard", vminmax=None, cmap=None, **render_opts_kwargs):

    opts = implicit_render_opts_object_from_args(**render_opts_kwargs)

    q = psb.render_implicit_surface_scalar(name, func, func_scalar, opts, str_to_datatype(datatype))

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
