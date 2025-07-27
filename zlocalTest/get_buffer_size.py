import tvm
from tvm.script import ir as I, tir as T
from tvm.tir import function
from tvm import relax
from tvm.relax.expr_functor import PyExprVisitor
from tvm.relax.expr import Call, Var, SeqExpr
from tvm.relax.analysis import post_order_visit


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

def get_shape_var_order(func: tvm.tir.PrimFunc):
    """提取 PrimFunc 中 shape 变量的顺序映射，例如 n → 0, m → 1"""
    order = {}
    idx = 0
    
    # 用于记录变量出现的顺序
    var_first_seen = {}
    
    def record_var(node):
        if isinstance(node, tvm.tir.Var) and node.dtype == "int32":
            if node not in var_first_seen:
                var_first_seen[node] = len(var_first_seen)
    
    # 1. 首先记录函数参数中的变量（按参数顺序）
    for param in func.params:
        if isinstance(param, tvm.tir.Var) and param.dtype == "int32":
            record_var(param)
    
    # 2. 遍历函数体，按出现顺序记录所有 int32 变量
    tvm.tir.stmt_functor.post_order_visit(func.body, record_var)
    
    # 3. 从 buffer_map 中记录 shape 变量
    for buffer in func.buffer_map.values():
        for dim in buffer.shape:
            tvm.tir.stmt_functor.post_order_visit(dim, record_var)
    
    # 4. 确定哪些变量是 shape 变量
    shape_vars = set()
    
    # 从 buffer shape 中收集
    for buffer in func.buffer_map.values():
        for dim in buffer.shape:
            if isinstance(dim, tvm.tir.Var) and dim.dtype == "int32":
                shape_vars.add(dim)
    
    # 从函数体中收集非参数的 int32 变量（很可能是 shape 变量）
    param_set = set(func.params)
    def collect_potential_shape_vars(node):
        if (isinstance(node, tvm.tir.Var) and 
            node.dtype == "int32" and 
            node not in param_set):
            shape_vars.add(node)
    
    tvm.tir.stmt_functor.post_order_visit(func.body, collect_potential_shape_vars)
    
    # 5. 按照第一次出现的顺序排序 shape 变量
    sorted_shape_vars = sorted(shape_vars, key=lambda v: var_first_seen.get(v, float('inf')))
    
    # 6. 分配顺序索引
    for var in sorted_shape_vars:
        order[var] = idx
        idx -= 1
    
    return order


def get_buffer_shape_list(buffer, var_order_dict):
    shape_repr = []
    dtype_size = tvm.runtime.DataType(buffer.dtype).bits // 8
    shape_repr.append(dtype_size)
    for dim in buffer.shape:
        if isinstance(dim, T.IntImm):
            shape_repr.append(int(dim.value))
        elif isinstance(dim, T.Var):
            shape_repr.append(var_order_dict.get(dim, str(dim)))
        else:
            shape_repr.append(str(dim))  # fallback
    return shape_repr


def get_buffer_shapes(mod, global_symbol: str):
    main_func = mod[global_symbol]
    num_info = extract_call_tir_info(main_func)
    new_functions = {}

    for gvar, func in mod.functions.items():
        if isinstance(func, function.PrimFunc):
            func_attrs = func.attrs
            target_attr = str(func_attrs.get("target").kind.name) if "target" in func_attrs else None

            if target_attr in ["fpga", "cuda"]:
                kernel_name = gvar.name_hint
                buffer_shapes = []
                arg_kinds = []
                slice_list = func_attrs.get("slice", [])
                slice_set = set(slice_list)

                if kernel_name in num_info:
                    input_num, _ = num_info[kernel_name]
                    var_order = get_shape_var_order(func)

                    buffer_shape_list = []
                    buffer_arg_kinds = []
                    scalar_shape_list = []
                    scalar_arg_kinds = []

                    for idx, param in enumerate(func.params):
                        if param in func.buffer_map:
                            buffer = func.buffer_map[param]
                            shape_list = get_buffer_shape_list(buffer, var_order)
                            buffer_shape_list.append(shape_list)

                            if param in slice_set:
                                buffer_arg_kinds.append(0)   # slice
                            elif idx >= input_num:
                                buffer_arg_kinds.append(3)   # output
                            else:
                                buffer_arg_kinds.append(1)   # input
                        else:
                            dtype_size = tvm.runtime.DataType(param.dtype).bits // 8
                            scalar_shape_list.append([dtype_size, 1])
                            scalar_arg_kinds.append(2)       # scalar or shape var

                    # 拼接顺序：buffer args + scalar args
                    buffer_shapes = buffer_shape_list + scalar_shape_list
                    arg_kinds = buffer_arg_kinds + scalar_arg_kinds

                new_func = func.with_attr("p2p_sizes", buffer_shapes)
                new_func = new_func.with_attr("arg_kinds", arg_kinds)
                new_functions[gvar] = new_func
            else:
                new_functions[gvar] = func
        else:
            new_functions[gvar] = func

    return tvm.IRModule(new_functions, global_infos=mod.global_infos)
