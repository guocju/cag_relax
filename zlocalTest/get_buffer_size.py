import tvm
from tvm.script import ir as I, tir as T
from tvm.tir import function
from tvm import relax

def extract_call_tir_info(func: relax.Function):
    """从 Relax IR 解析 call_tir 语句的 kernel_name, input_num, out_num"""
    @relax.expr_functor.visitor
    class CallTIRVisitor(tvm.relax.PyExprVisitor):
        def __init__(self):
            super().__init__()
            self.call_tir_infos = {}

        def visit_call_(self, call):
            if isinstance(call.op, tvm.ir.op.Op) and call.op.name == "relax.call_tir":
                kernel_name = call.args[0].name_hint
                input_num = len(call.args[1])
                out_num = len(call.checked_type.fields) if isinstance(call.checked_type, tvm.ir.type.TupleType) else 1
                self.call_tir_infos[kernel_name] = (input_num, out_num)
            super().visit_call_(call)

    visitor = CallTIRVisitor()
    visitor.visit_expr(func.body)
    return visitor.call_tir_infos


def get_buffer_sizes(mod, global_symbol: str):
    main_func = mod[global_symbol]
    num_info = extract_call_tir_info(main_func)
    new_functions = {}
    for Global_Var, func in mod.functions.items():
        if isinstance(func, function.PrimFunc):  # 只筛选 TIR 函数
            func_attrs = func.attrs
            if func_attrs and "target" in func_attrs and str(func_attrs["target"].kind.name) == "fpga":
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
                new_functions[Global_Var] = new_func
            elif func_attrs and "target" in func_attrs and str(func_attrs["target"].kind.name) == "cuda":
                func_name = Global_Var.name_hint
                if func_name[:3] == "p2p":
                    input_num, out_num = num_info[func_name]
                    buffer_sizes = []
                    for idx, param in enumerate(func.params):
                        if idx >= input_num:
                            break
                        buffer = func.buffer_map[param]
                        buffer_name = buffer.name
                        if len(buffer_name) == 1 and 'A' <= buffer_name <= 'Z':
                            dtype_size = tvm.runtime.DataType(buffer.dtype).bits // 8 # use byte
                            num_elements = 1
                            for dim in buffer.shape:
                                num_elements *= int(dim)
                            buffer_sizes.append(num_elements * dtype_size)
                        
                    new_func = func.with_attr("p2p_sizes", buffer_sizes)
                    new_functions[Global_Var] = new_func
                else:
                    new_functions[Global_Var] = func
            else:
                new_functions[Global_Var] = func
        else:
            new_functions[Global_Var] = func  # 非 PrimFunc 保持不变

    new_mod = tvm.IRModule(new_functions)
    return new_mod