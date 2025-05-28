import memray
from contextlib import nullcontext
from line_profiler import profile as line_profile
from memory_profiler import profile as memory_profile

class base_profile_manager:
    def __init__(self, rank):
        self.rank = rank
class line_profile_manager(base_profile_manager):
    def __enter__(self):
        line_profile.enable(output_prefix=f'line_profile-{self.rank}')
        return line_profile

    def __exit__(self, exc_type, exc_val, exc_tb):
        line_profile.disable()


class memory_profile_manager(base_profile_manager):
    def __enter__(self):
        self.fp = open(f"memory_profile-{self.rank}", "w")
        return lambda x: memory_profile(x, stream=self.fp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fp.close()


class memray_profile_manager(base_profile_manager):
    def __enter__(self):
        self._destination = memray.FileDestination(
            f"memray-{self.rank}.bin", overwrite=True, compress_on_exit=True
        )
        self._tracker = memray.Tracker(
            destination=self._destination,
            native_traces=True,
            trace_python_allocators=True,
            memory_interval_ms=5
        )
        self._tracker.__enter__()
        return lambda x: x  # return identity function

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tracker.__exit__(exc_type, exc_val, exc_tb)

def profiler_manager(profile_type, rank):
    if profile_type == "line_profile":
        return line_profile_manager(rank)
    if profile_type == "memory_profile":
        return memory_profile_manager(rank)
    if profile_type == "memray":
        return memray_profile_manager(rank)
    return nullcontext(lambda x : x)