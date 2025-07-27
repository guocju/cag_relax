import tvm
import numpy as np
from tvm import relax
from tvm.script import relax as R, tir as T, ir as I
from tvm.relay.backend import Runtime

n = 8
slicelen = 2
target = tvm.target.Target("llvm")
host = tvm.target.Target("c")
runtime = Runtime("crt", {"system-lib": True})

@I.ir_module
class Testpipeline:
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                ]
            }
        )

    @T.prim_func
    def soft_pipeline_bank4(
        x: T.handle,
        slicelen: T.int32,
        y: T.handle,
    ):
        T.func_attr({"global_symbol": "soft_pipeline_bank4", "tir.noalias": True, "target": target})

        n = T.int32()
        X = T.match_buffer(x, (n,), dtype="int32")
        Y = T.match_buffer(y, (n,), dtype="int32")
        
        # 4-bank double buffer
        lm_in = T.alloc_buffer([4, 1024], "int32", scope="local")
        lm_out = T.alloc_buffer([4, 1024], "int32", scope="local")
        num_blocks = T.floordiv(n + slicelen - 1, slicelen)
        for bo in range(0, num_blocks + 2):
            # with T.block("soft_pipeline"):
            #     T.block_attr({
            #         "software_pipeline_stage": [0, 1, 2],
            #         "software_pipeline_order": [0, 1, 2],
            #         "software_pipeline_async_stages": [0],
            #     })

                # Stage 0: load
                if bo < num_blocks:
                    for i in range(T.min(slicelen, n - bo * slicelen)):
                        lm_in[T.floormod(bo, 4), i] = X[bo * slicelen + i]

                # Stage 1: compute
                if (bo > 0) and (bo <= num_blocks):
                    for i in range(T.min(slicelen, n - (bo - 1) * slicelen)):
                        lm_out[T.floormod(bo - 1, 4), i] = lm_in[T.floormod(bo - 1, 4), i] + 1

                # Stage 2: store
                if (bo > 1) and (bo <= num_blocks + 1):
                    for i in range(T.min(slicelen, n - (bo - 2) * slicelen)):
                        Y[(bo - 2) * slicelen + i] = lm_out[T.floormod(bo - 2, 4), i]

    @R.function
    def main(x: R.Tensor(("n",), dtype="int32")):
        n = T.int64()  # dynamic shape
        slicelength = R.prim_value(T.IntImm("int32", slicelen))
        with R.dataflow():
            y = R.call_tir(Testpipeline.soft_pipeline_bank4, (x, slicelength), out_sinfo=R.Tensor((n,), dtype="int32"))
            R.output(y)
        return y
                


mod = Testpipeline
target = tvm.target.Target("c")
ex = relax.build(mod, target)

ex_tir = tvm.build(mod["soft_pipeline_bank4"], target=host, runtime=runtime)
imported_module = ex_tir.imported_modules[0]
with open("zlocalTest/p2p_auto_test/codegen.c", "w") as f:
    f.write(imported_module.get_source("c"))

# Create VM runtime
dev = tvm.cpu()
vm = relax.VirtualMachine(ex, dev)

# Test input
x_np = np.arange(n, dtype="int32")
x_nd = tvm.nd.array(x_np, dev)

# Run
res = vm["main"](x_nd)

# Check
y_np = x_np + 1
np.testing.assert_allclose(res.numpy(), y_np)
print("测试通过！输出正确。")
