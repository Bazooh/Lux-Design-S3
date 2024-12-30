import gc
import jax
def reset_device_memory(delete_objs=True):
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers(): buf.delete()