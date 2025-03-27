import tvm
from tvm.script import ir as I, tir as T
from tvm.tir import function

def get_buffer_sizes(mod):
    new_functions = {}
    for name, func in mod.functions.items():
        if isinstance(func, function.PrimFunc):  # 只筛选 TIR 函数
            func_attrs = func.attrs
            if func_attrs and "target" in func_attrs and str(func_attrs["target"]) == "fpga ":
                param_num = func_attrs["param_num"]
                buffer_sizes = []
                i = 0
                for param in func.params:
                    i = i+1
                    if 1 < i < param_num+2:
                        continue
                    buffer = func.buffer_map[param]
                    dtype_size = tvm.runtime.DataType(buffer.dtype).bits // 8 # use byte
                    num_elements = 1
                    for dim in buffer.shape:
                        num_elements *= int(dim)
                    buffer_sizes.append(num_elements * dtype_size)
                    
                new_func = func.with_attr("p2p_sizes", buffer_sizes)
                new_functions[name] = new_func
            else:
                new_functions[name] = func
        else:
            new_functions[name] = func  # 非 PrimFunc 保持不变

    new_mod = tvm.IRModule(new_functions)
    return new_mod