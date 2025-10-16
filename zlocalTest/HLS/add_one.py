import tvm
from tvm import tir
from tvm.script import tir as T
import numpy as np

def get_task_c_code(task_type:str):
    # 将提前注册的任务类型映射到用于HLS的C代码路径，返回该路径的字符串
    return

def generate_circuit(code_path):
    # 使用c代码进行HLS电路生成和综合，产生bit文件
    return

def hls_build(task_type:str, module):
    code_path = get_task_c_code(task_type)
    generate_circuit(code_path)
    rt_mod = tvm.build(module, target="llvm")
    return rt_mod

def programm_device(task_type:str):
    # 将bit烧录到fpga
    success: bool = True
    return success

def prepare_rvcode(task_type:str):
    # 准备rvcode，将任务对应的rv sw_code.bin搬运到p2p_demo/riscv/output/bin，按照约定工具链将rv bin文件放进fpga里面
    return

fpga = tvm.fpga()

# -------------------------------
# Python端：注册 PackedFunc
# -------------------------------
@tvm.register_func("my_add_one")
def my_add_one(A_nd: tvm.nd.NDArray, x, B_nd: tvm.nd.NDArray, n):
    print(">>> Python 函数 my_add_one 被执行")
    success = programm_device("add_one")
    prepare_rvcode("add_one")
    # copyto会在fpga ddr malloc一块内存，并调用xdma进行传输
    A_fpga = A_nd.copyto(fpga)
    B_fpga = B_nd.copyto(fpga)
    f = tvm.get_global_func("vm.builtin.fpga.launch_kernel")
    # kernel launch会将rv bin复制到fpga内部，完成对rv的配置，参数下发到riscv，riscv内部相应的代码会解析这些参数，注意顺序一致
    f(A_fpga, B_fpga, n, x)
    # 先进行同步，等待fpga任务完成，调用xdma传递给host
    B_nd.copyfrom(B_fpga)

@tvm.script.ir_module
class AddOneModule:
    @T.prim_func
    def main(a_handle: T.handle,
             x: T.int32,
             b_handle: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        n = T.int32()
        A = T.match_buffer(a_handle, (n,), dtype="uint8")
        B = T.match_buffer(b_handle, (n,), dtype="uint8")
        T.call_packed("my_add_one", A, x, B, n)


# -------------------------------
# Build 阶段
# -------------------------------
ir_mod = AddOneModule
print(ir_mod)
rt_mod = hls_build("add_one", ir_mod)

# -------------------------------
# 运行测试
# -------------------------------
A_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="uint8")
x = 5
B_np = np.zeros_like(A_np)

# 封装为 NDArray
A_tvm = tvm.nd.array(A_np)
B_tvm = tvm.nd.array(B_np)

# 执行 module
rt_mod(A_tvm, x, B_tvm)

print("输入 A =", A_np)
print("常数 x =", x)
print("输出 B =", B_tvm.numpy())
print("期望结果 =", (A_np + x).astype("uint8"))
