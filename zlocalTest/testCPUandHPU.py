import tvm
from tvm import relay
from tvm.relay import transform
import onnx
from tvm import runtime
import numpy as np
import sys
import os


third_party_path = os.path.abspath("/home/guocj/cag_relax/3rdparty/HPU500/kernel_library_hpu500/script/compiler")
sys.path.append(third_party_path)
from compiler.frontend.pattern_match.PatternSetQuant import pattern_table_quant, hyper_pattern_table
from compiler.frontend.pattern_match.PatternSetFloat import pattern_table_float, partition_mode

class CallNodeRewriter(relay.ExprMutator):
    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        if isinstance(call.op, relay.GlobalVar):
            new_call = relay.annotation.on_device(call, "fpga")
            return new_call
        return relay.Call(call.op, new_args, call.attrs)
    
class CallNodeArgsRewriter(relay.ExprMutator):
    def arg_on_device(self, call):
        new_arg = relay.annotation.on_device(call, "cpu")
        return new_arg
        
    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        if isinstance(call.op, relay.GlobalVar):
            new_args = [self.arg_on_device(arg) for arg in call.args]
        return relay.Call(call.op, new_args, call.attrs)
    
def modify_call_nodes_in_module(mod):
    function_rewriter = CallNodeRewriter()
    args_rewriter = CallNodeArgsRewriter()
    new_mod = tvm.IRModule()

    for global_var, func in mod.functions.items():
        if global_var.name_hint == "main":
            new_func = function_rewriter.visit(func)
        else:
            new_func = func
        new_mod[global_var] = new_func
        
    for global_var, func in new_mod.functions.items():
        if global_var.name_hint == "main":
            new_func = args_rewriter.visit(func)
        else:
            new_func = func
        new_mod[global_var] = new_func

    return new_mod


# 加载 ONNX 模型
onnx_model_path = "/home/guocj/env/f07b_zu9p_prj/algorithm/data_param_20240528/resnet18_modified.onnx"
onnx_model = onnx.load(onnx_model_path)

# 定义输入形状
input_name = "input"
input_shape = (1, 3, 96, 96)
input_dtype = "float32"


# 转换为 Relay IR
mod, params = relay.frontend.from_onnx(onnx_model)
pattern1_quant, pattern2_quant = pattern_table_quant()
mod = partition_mode(mod, pattern1_quant)
mod = partition_mode(mod, pattern2_quant)

pattern1_float, pattern2_float, pattern3_float = pattern_table_float()
mod = partition_mode(mod, pattern1_float)
mod = partition_mode(mod, pattern2_float)
mod = relay.transform.AnnotateTarget("hpu")(mod)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)
mod = modify_call_nodes_in_module(mod)

HOST_DEVICE = tvm.device("cpu")
HOST_TARGET = tvm.target.Target("llvm")

CPU_DEVICE = tvm.device("cpu")
CPU_TARGET = tvm.target.Target("llvm").with_host(HOST_TARGET)

FPGA_DEVICE = tvm.device("fpga")
FPGA_TARGET = tvm.target.Target("fpga").with_host(HOST_TARGET)

TARGETS = [CPU_TARGET, FPGA_TARGET]
CTXT = tvm.transform.PassContext(config={"relay.fallback_device_type": tvm.cpu().device_type})
config = tvm.target.make_compilation_config(CTXT, TARGETS)
mod = relay.transform.InferType()(mod)
mod = relay.transform.PlanDevices(config)(mod)
mod = relay.transform.InferType()(mod)


with tvm.transform.PassContext(
    opt_level=0, config={"relay.fallback_device_type": tvm.cpu().device_type}
):
    exe = relay.vm.compile(
        mod, target={"cpu": tvm.target.Target("llvm"), "hpu": tvm.target.Target("fpga")}
    )

# Run
vm = runtime.vm.VirtualMachine(exe, [tvm.cpu(), tvm.fpga()])
input_data = np.random.rand(*input_shape).astype(input_dtype)
input_data = tvm.nd.array(input_data, device=tvm.cpu())
actual_result = vm.invoke("main", input_data)