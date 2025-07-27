# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_global_infos({"vdevice": [I.vdevice({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-pc-linux-gnu", "tag": ""}, 0, "global")]})
    @T.prim_func
    def soft_pipeline_bank4(x: T.handle, slicelen: T.int32, y: T.handle):
        T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-pc-linux-gnu", "tag": ""}), "tir.noalias": T.bool(True)})
        n = T.int32()
        X = T.match_buffer(x, (n,), "int32")
        Y = T.match_buffer(y, (n,), "int32")
        lm_in = T.allocate([4096], "int32", "local")
        lm_out = T.allocate([4096], "int32", "local")
        Module.soft_pipeline_bank4_kernel_1(X.data, Y.data, lm_in, lm_out, n, slicelen)

    @T.prim_func(private=True)
    def soft_pipeline_bank4_kernel_1(X: T.handle("int32", "global"), Y: T.handle("int32", "global"), lm_in: T.handle("int32", "local"), lm_out: T.handle("int32", "local"), n: T.int32, slicelen: T.int32):
        T.func_attr({"target": T.target({"keys": [], "kind": "fpga", "tag": ""}), "tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        Y_1 = T.decl_buffer((n,), "int32", data=Y)
        lm_out_1 = T.decl_buffer((4096,), "int32", data=lm_out, scope="local")
        X_1 = T.decl_buffer((n,), "int32", data=X)
        lm_in_1 = T.decl_buffer((4096,), "int32", data=lm_in, scope="local")
        for bo in range((n + slicelen - 1) // slicelen + 2):
            if bo < (n + slicelen - 1) // slicelen:
                for i in range(T.min(slicelen, n - bo * slicelen)):
                    lm_in_1[T.max(0, bo % 4 - (bo + 3) % 4) * 1024 + i] = X_1[bo * slicelen + i]
            if 0 < bo and bo <= (n + slicelen - 1) // slicelen:
                for i in range(T.min(slicelen, n - (bo - 1) * slicelen)):
                    cse_var_1: T.int32 = (bo + 3) % 4
                    lm_out_1[T.max(0, cse_var_1 - (bo + 2) % 4) * 1024 + i] = lm_in_1[T.max(cse_var_1 - bo % 4, 0) * 1024 + i] + 1
            if 1 < bo:
                for i in range(T.min(slicelen, n - (bo - 2) * slicelen)):
                    Y_1[(bo - 2) * slicelen + i] = lm_out_1[T.max((bo + 2) % 4 - (bo + 3) % 4, 0) * 1024 + i]

    @R.function
    def main(x: R.Tensor(("n",), dtype="int32")) -> R.Tensor(("n",), dtype="int32"):
        n = T.int64()
        cls = Module
        slicelength: R.Prim(value=2) = R.prim_value(2)
        with R.dataflow():
            y = R.call_tir(cls.soft_pipeline_bank4, (x, slicelength), out_sinfo=R.Tensor((n,), dtype="int32"))
            R.output(y)
        return y