"""GDN-only ablation: substitute packed decode inside this lab worker process."""
from vllm.v1.worker.gpu_worker import Worker


class GDNWorker(Worker):
    def init_device(self):
        super().init_device()
        from gdn_pass.kernel import load,dispatch
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as module
        load()
        module.fused_recurrent_gated_delta_rule_packed_decode=dispatch
        print('QWEN_GDN_LAB: packed decode CUDA replacement installed',flush=True)
