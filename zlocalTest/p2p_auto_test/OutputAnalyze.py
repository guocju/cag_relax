import tvm
from tvm.script import ir as I, tir as T
from tvm.tir import function
from tvm import relax
from tvm.relax.expr_functor import PyExprVisitor
from tvm.relax.expr import Call, Var, SeqExpr
from tvm.relax.analysis import post_order_visit

target = tvm.target.Target("fpga")

def run_passes(mod):
    # mod = tvm.driver.build_module.lower(mod)
    # mod = tvm.tir.transform.BindTarget(target)(mod)
    # mod = tvm.tir.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tir.transform.SplitHostDevice()(mod)
    return mod

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


def get_buffer_shapes(mod, global_symbol: str):
    main_func = mod[global_symbol]
    num_info = extract_call_tir_info(main_func)
    new_functions = {}

    FPGA_DTYPE_MAP = {
        "uint8": 0, "uint16": 1, "uint32": 2, "uint64": 3,
        "int8": 4, "int16": 5, "int32": 6, "int64": 7,
        "float16": 8, "float32": 9, "float64": 10,
        "handle": 11,
    }

    HANDLE_BASE = FPGA_DTYPE_MAP["handle"]

    for gvar, func in mod.functions.items():
        if isinstance(func, function.PrimFunc):
            func_attrs = func.attrs
            target_attr = str(func_attrs.get("target").kind.name) if "target" in func_attrs else None

            if target_attr == "fpga":
                kernel_name = gvar.name_hint
                buffer_shapes = []
                arg_dtypes = []

                if kernel_name in num_info:
                    # 构建 shape var → 负 index 映射
                    ir_mod = tvm.IRModule.from_expr(func)
                    ir_mod = run_passes(ir_mod)
                    kernel_func = ir_mod[kernel_name+"_kernel_1"]
                    param_index_map = {param: -i for i, param in enumerate(kernel_func.params)}

                    for param in kernel_func.params:
                        if param in kernel_func.buffer_map:
                            # 是 buffer 类型参数
                            buffer = kernel_func.buffer_map[param]
                            dtype = buffer.dtype.lower()
                            if dtype not in FPGA_DTYPE_MAP:
                                raise ValueError(f"Unsupported dtype: {dtype}")
                            ptr_dtype_code = HANDLE_BASE + FPGA_DTYPE_MAP[dtype]
                            arg_dtypes.append(ptr_dtype_code)

                            shape_repr = [tvm.runtime.DataType(dtype).bits // 8]
                            for dim in buffer.shape:
                                if isinstance(dim, T.IntImm):
                                    shape_repr.append(int(dim.value))
                                elif isinstance(dim, tvm.tir.expr.Var):
                                    shape_repr.append(param_index_map.get(dim, str(dim)))
                                else:
                                    shape_repr.append(str(dim))
                            buffer_shapes.append(shape_repr)
                        else:
                            # 是非 buffer 的 scalar 参数（shape var or const）
                            dtype = param.dtype.lower()
                            if dtype not in FPGA_DTYPE_MAP:
                                raise ValueError(f"Unsupported dtype: {dtype}")
                            arg_dtypes.append(FPGA_DTYPE_MAP[dtype])
                            buffer_shapes.append([tvm.runtime.DataType(dtype).bits // 8, 1])

                new_func = func.with_attr("p2p_sizes", buffer_shapes)
                new_func = new_func.with_attr("arg_kinds", arg_dtypes)
                new_functions[gvar] = new_func
            else:
                new_functions[gvar] = func
        else:
            new_functions[gvar] = func

    return tvm.IRModule(new_functions, global_infos=mod.global_infos)


