import polyscope_bindings as psb

import os

import numpy as np

from polyscope.core import get_render_engine_backend_name


# Try both of these imports, but fail silently if they don't work
# (we will try again and print an informative error bleow only if
# a relevant function is called)
try: 
    import cuda
    from cuda import cudart
except ImportError: 
    pass
try: 
    import cupy
except ImportError: 
    pass

## Experimental things

def check_device_module_availibility():

    supported_backends = ["openGL3_glfw"]
    if get_render_engine_backend_name() not in supported_backends:
        raise ValueError(f"This Polyscope functionality is not supported by the current rendering backend ({get_render_engine_backend_name()}. Supported backends: {','.join(supported_backends)}.")

    try:
        import cuda
    except ImportError:
        raise ImportError('This Polyscope functionality requires cuda bindings to be installed: https://nvidia.github.io/cuda-python/')

    try:
        import cupy
    except ImportError:
        raise ImportError('This Polyscope functionality requires cupy to be installed: https://cupy.dev/')


# TODO is it possible to implement this without relying on exceptions?
'''
def is_dlpack(obj):
    return hasattr(obj, '__dlpack__') and hasattr(obj, '__dlpack_device__')
'''

def is_cuda_array_interface(obj):
    return hasattr(obj, '__cuda_array_interface__')


def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


# A helper class to manage openGL buffers which are mapped for direct writing from GPU data

# A global cache of device buffers below
# (keyed on the structure/quantity unique name)
mapped_device_buffers = {}

def get_mapped_buffer(key, buff):
    # if we have not already mapped the buffer, do it now
    if key in mapped_device_buffers: 

        mapped_buff = mapped_device_buffers[key]
        if mapped_buff.is_same_buffer(buff):
            return mapped_buff

    # map a new buffer
    mapped_buff = CudaOpenGLMappedBuffer(buff)
    mapped_device_buffers[key] = mapped_buff

    return mapped_buff

class CudaOpenGLMappedBuffer:

    # More goodies here: https://gist.github.com/keckj/e37d312128eac8c5fca790ce1e7fc437

    def __init__(self, gl_attribute_buffer):
        self.gl_attribute_buffer = gl_attribute_buffer
        self.resource_handle = None
        self.cuda_buffer_ptr = None
        self.cuda_buffer_size = -1
        self.mapped = False

        # Register the buffer 
        self.resource_handle = check_cudart_err(
            cudart.cudaGraphicsGLRegisterBuffer(
                self.gl_attribute_buffer.get_native_buffer_id(), 
                cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone
                )
        )

    def unregister(self):
        self.unmap()
        self.resource_handle = check_cudart_err(
            cudart.cudaGraphicsUnregisterResource(self.resource_handle))

    def is_same_buffer(self, other_gl_attribute_buffer):
        return self.gl_attribute_buffer == other_gl_attribute_buffer

    def map(self):
        """
        Returns a cupy memory pointer to the buffer
        """

        if self.cuda_buffer_ptr is not None:
                return self.cuda_buffer_ptr

        check_cudart_err(cudart.cudaGraphicsMapResources(1, self.resource_handle, None))

        ptr, size = check_cudart_err(cudart.cudaGraphicsResourceGetMappedPointer(self.resource_handle))

        self.cuda_buffer_ptr = cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(ptr, size, self), 0)
        self.cuda_buffer_size = size

        return self.cuda_buffer_ptr

    def unmap(self):
        if self.cuda_buffer_ptr is None:
            return

        check_cudart_err(cudart.cudaGraphicsUnmapResources(1, self.resource_handle, None))

        self.cuda_buffer_ptr = None
        self.cuda_buffer_size = -1
    
    def set_data_from_array(self, arr, expected_shape, expected_dtype):
            
        # Dispatch to one of the kinds of objects that we can read from
        
        # __cuda_array_interface__
        if is_cuda_array_interface(arr):
            self.set_data_from_array_cuda_array_interface(arr, expected_shape, expected_dtype)
            return

        # __dlpack__
        # (can't figure out any way to check this except try-catch)
        try:
            self.set_data_from_array_dlpack(arr, expected_shape, expected_dtype)
            return
        except ValueError:
            pass 
       


        raise ValueError("Cannot read from device data object. Must be a _dlpack_ array or implement the __cuda_array_interface__.")

    def set_data_from_array_dlpack(self, arr, expected_shape, expected_dtype):

        self.map()
       
        cupy_arr = cupy.ascontiguousarray(cupy.from_dlpack(arr))

        # do some shape & type checks
        # (we intentionally return a RuntimeError here, because cupy returns a ValueError when its not __dl_pack__ and we use that above)
        if cupy_arr.dtype != expected_dtype:
            raise RuntimeError(f"dlpack array has wrong dtype, expected {expected_dtype} but got {cupy_arr.dtype}")
        if cupy_arr.shape != expected_shape:
            raise RuntimeError(f"dlpack array has wrong shape, expected {expected_shape} but got {cupy_arr.shape}")
        if cupy_arr.nbytes != self.cuda_buffer_size: 
            # if cupy_arr has the right size/dtype, it should have exactly the same 
            # number of bytes as the destination. This is just lazily saving us 
            # from repeating the math, and also directly validates the copy we 
            # are about to do below.
            raise RuntimeError(f"mapped buffer has wrong size, expected {cupy_arr.nbytes} bytes but got {self.cuda_buffer_size}. (This is probably an internal Polyscope problem, not a problem with the passed array.")
        
        self.cuda_buffer_ptr.copy_from_device(cupy_arr.data, self.cuda_buffer_size)

        self.unmap() 

    def set_data_from_array_cuda_array_interface(self, arr, expected_shape, expected_dtype):

        self.map()
       
        cupy_arr = cupy.ascontiguousarray(cupy.asarray(arr))

        # do some shape & type checks
        # (we intentionally return a RuntimeError here, because cupy returns a ValueError when its not __dl_pack__ and we use that above)
        if cupy_arr.dtype != expected_dtype:
            raise RuntimeError(f"dlpack array has wrong dtype, expected {expected_dtype} but got {cupy_arr.dtype}")
        if cupy_arr.shape != expected_shape:
            raise RuntimeError(f"dlpack array has wrong shape, expected {expected_shape} but got {cupy_arr.shape}")
        if cupy_arr.nbytes != self.cuda_buffer_size: 
            # if cupy_arr has the right size/dtype, it should have exactly the same 
            # number of bytes as the destination. This is just lazily saving us 
            # from repeating the math, and also directly validates the copy we 
            # are about to do below.
            raise RuntimeError(f"mapped buffer has wrong size, expected {cupy_arr.nbytes} bytes but got {self.cuda_buffer_size}. (This is probably an internal Polyscope problem, not a problem with the passed array.")
        
        self.cuda_buffer_ptr.copy_from_device(cupy_arr.data, self.cuda_buffer_size)

        self.unmap() 
