"""Explicit worker selection confines the experiment to the lab engine."""
from vllm.v1.worker.gpu_worker import Worker


class KernelWorker(Worker):
    def init_device(self):
        super().init_device()
        from custom_kernels.integration import install
        install()
