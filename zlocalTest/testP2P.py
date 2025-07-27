import os
import tvm
import numpy as np
import tvm.script
import tvm.testing
from tvm import relax, tir
from tvm.script import relax as R, tir as T, ir as I
from tvm.ir.global_info import VDevice
from typing import List
from tvm._ffi.runtime_ctypes import Device
from tvm.ir.module import IRModule

def compile(
    mod: IRModule,
    device: List[Device] = [
        tvm.cpu(), tvm.fpga(), tvm.cuda()
    ],
) -> relax.VirtualMachine:
    # compile the model
    mod = relax.transform.RealizeVDevice()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    # no need to feed target argument for mult-target compilation
    with tvm.transform.PassContext(
        opt_level=0,
        config={"tir.disable_storage_rewrite": True},
    ):
        ex = relax.build(mod)
    return relax.VirtualMachine(ex, device, buffer_num=2)


llvm_target = tvm.target.Target("llvm")
fpga_target = tvm.target.Target("fpga")
cuda_target = tvm.target.Target("cuda")
slice_number = 46
val_dtype = "float32"
thread_extent_for_short = 128
thread_extent_for_long = 128
length_threshold = 32
long_align_val = 16
short_align_val = 8
ele_num = 5000
ROW = 100
COL = 100

@I.ir_module
class TestP2P:
    I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("cuda", 0),
                    I.vdevice("fpga", 0),
                ]
            }
        )
    # data, params, output_buffers, other
    @T.prim_func
    def fpga_process(
        indptr: T.handle,
        indices: T.handle,
        data: T.handle,
        # B: T.Buffer((4,), "int8"),
        row_idx_s: T.handle,
        col_idx_s: T.handle,
        val_s: T.handle,
        row_idx_l: T.handle,
        col_idx_l: T.handle,
        val_l: T.handle,
    ) -> None:
        T.func_attr({"target": T.target("fpga"), "func_type": "preprocess"})
        row_add_one = T.int32()
        nnz4all = T.int32()
        indptr_buffer = T.match_buffer(indptr, (row_add_one,), "int32")
        indices_buffer = T.match_buffer(indices, (nnz4all,), "int32")
        data_buffer = T.match_buffer(data, (nnz4all,), val_dtype)
        row_s = T.int64()
        row_l = T.int64()

        row_idx_s_buffer = T.match_buffer(row_idx_s, (row_s,), "int32")
        col_idx_s_buffer = T.match_buffer(col_idx_s, (short_align_val, row_s), "int32")
        val_s_buffer = T.match_buffer(val_s, (short_align_val, row_s), val_dtype)
        status_s = T.alloc_buffer((2,), "int32")

        row_idx_l_buffer = T.match_buffer(row_idx_l, (row_l,), "int32")
        col_idx_l_buffer = T.match_buffer(col_idx_l, (row_l, long_align_val), "int32")
        val_l_buffer = T.match_buffer(val_l, (row_l, long_align_val), val_dtype)
        status_l = T.alloc_buffer((2,), "int32")

        T.attr(T.target("fpga"), "target", 0)
        row: T.int32 = row_add_one - 1
        with T.block("initialize"):
            status_s[0] = 0
            status_s[1] = 0
            status_l[0] = 0
            status_l[1] = 0

        with T.block("compute"):
            for row_idx in T.serial(row):
                nnz = indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]
                if nnz <= T.int32(length_threshold):
                    for idx in T.serial(nnz):
                        from_idx: T.int32 = indptr_buffer[row_idx] + idx
                        if idx % T.int32(short_align_val) == 0:
                            row_idx_s_buffer[status_s[0]] = row_idx
                            status_s[0] += 1

                        x: T.int32 = T.floormod(status_s[1], T.int32(short_align_val))
                        y: T.int32 = T.floordiv(status_s[1], T.int32(short_align_val))
                        col_idx_s_buffer[x, y] = indices_buffer[from_idx]
                        val_s_buffer[x, y] = data_buffer[from_idx]
                        status_s[1] += 1

                    if status_s[1] % T.int32(short_align_val) != 0:
                        for i in T.serial(
                            T.int32(short_align_val)
                            - T.floormod(status_s[1], T.int32(short_align_val))
                        ):
                            x: T.int32 = T.floormod(status_s[1], T.int32(short_align_val))
                            y: T.int32 = T.floordiv(status_s[1], T.int32(short_align_val))
                            col_idx_s_buffer[x, y] = 0
                            val_s_buffer[x, y] = 0
                            status_s[1] += 1

                else:
                    for idx in T.serial(nnz):
                        from_idx: T.int32 = indptr_buffer[row_idx] + idx
                        if idx % T.int32(long_align_val) == 0:
                            row_idx_l_buffer[status_l[0]] = row_idx
                            status_l[0] += 1

                        x: T.int32 = T.floordiv(status_l[1], T.int32(long_align_val))
                        y: T.int32 = T.floormod(status_l[1], T.int32(long_align_val))
                        col_idx_l_buffer[x, y] = indices_buffer[from_idx]
                        val_l_buffer[x, y] = data_buffer[from_idx]
                        status_l[1] += 1

                    if status_l[1] % T.int32(long_align_val) != 0:
                        for i in T.serial(
                            T.int32(long_align_val)
                            - T.floormod(status_l[1], T.int32(long_align_val))
                        ):
                            x: T.int32 = T.floordiv(status_l[1], T.int32(long_align_val))
                            y: T.int32 = T.floormod(status_l[1], T.int32(long_align_val))
                            col_idx_l_buffer[x, y] = 0
                            val_l_buffer[x, y] = 0
                            status_l[1] += 1
    
    @T.prim_func
    def long_lines(value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle):
        T.func_attr({"target": cuda_target})
        row_num_split, col_num_split = T.int32(), T.int32()
        buffer_1_val = T.match_buffer(value_1, (row_num_split, col_num_split))
        col_1_idx = T.match_buffer(c_idx_1, (row_num_split, col_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        col_num_real = T.int32()
        vec_val = T.match_buffer(vector_in, (col_num_real,))
        row_num_real = T.int32()
        res_val = T.match_buffer(res_t, (row_num_real,))
        # with T.block("root"):
        if row_num_split > 0:
            with T.block("outer"):
                T.reads(buffer_1_val[0:row_num_split, 0:col_num_split], vec_val[0:col_num_real], col_1_idx[0:row_num_split, 0:col_num_split], res_val[0:row_num_real], buffer_1_idx[0:row_num_split])
                T.writes(res_val[0:row_num_real])
                for real_row_idx_0 in T.thread_binding((row_num_real + 255) // 256, thread="blockIdx.x"):
                    for real_row_idx_1 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("spmv_init"):
                            v_real_row_idx = T.axis.spatial(row_num_real, real_row_idx_0 * 256 + real_row_idx_1)
                            T.where(real_row_idx_0 * 256 + real_row_idx_1 < row_num_real)
                            T.reads()
                            T.writes(res_val[v_real_row_idx])
                            res_val[v_real_row_idx] = T.float32(0.0)
                for row_idx in T.thread_binding(row_num_split, thread="blockIdx.x"):
                    with T.block("spmv_long_o"):
                        v_row_idx = T.axis.spatial(row_num_split, row_idx)
                        T.reads(buffer_1_val[v_row_idx, 0:col_num_split], vec_val[0:col_num_real], col_1_idx[v_row_idx, 0:col_num_split], res_val[buffer_1_idx[v_row_idx]], buffer_1_idx[v_row_idx])
                        T.writes()
                        res_local = T.alloc_buffer((1,), scope="shared")
                        for col_idx_0 in range((col_num_split + 127) // 128):
                            for col_idx_1 in T.thread_binding(128, thread="threadIdx.x"):
                                with T.block("spmv_long"):
                                    v_col_idx = T.axis.reduce(col_num_split, col_idx_0 * 128 + col_idx_1)
                                    T.where(col_idx_0 * 128 + col_idx_1 < col_num_split)
                                    T.reads(buffer_1_val[v_row_idx, v_col_idx], vec_val[col_1_idx[v_row_idx, v_col_idx]], col_1_idx[v_row_idx, v_col_idx])
                                    T.writes(res_local[0])
                                    with T.init():
                                        res_local[0] = T.float32(0.0)
                                    res_local[0] = res_local[0] + buffer_1_val[v_row_idx, v_col_idx] * vec_val[col_1_idx[v_row_idx, v_col_idx]]
                        tx = T.launch_thread("threadIdx.x", 128)
                        with T.block("spmv_long_2"):
                            T.reads(res_val[buffer_1_idx[v_row_idx]], buffer_1_idx[v_row_idx], res_local[0])
                            T.writes()
                            T.tvm_storage_sync("shared")
                            if tx == 0:
                                T.call_extern("float32", "atomicAdd", T.address_of(res_val[buffer_1_idx[v_row_idx]]), res_local[0])
        else:
            with T.block("outer"):
                T.reads()
                T.writes(res_val[0:row_num_real])
                for real_row_idx_0 in T.thread_binding((row_num_real + 255) // 256, thread="blockIdx.x"):
                    for real_row_idx_1 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("spmv_init_2"):
                            v_real_row_idx = T.axis.spatial(row_num_real, real_row_idx_0 * 256 + real_row_idx_1)
                            T.where(real_row_idx_0 * 256 + real_row_idx_1 < row_num_real)
                            T.reads()
                            T.writes(res_val[v_real_row_idx])
                            res_val[v_real_row_idx] = T.float32(0.0)

    @T.prim_func
    def short_lines_easier(value_1: T.handle, c_idx_1: T.handle, idx_1: T.handle, vector_in: T.handle, res_t: T.handle):
        T.func_attr({"target": cuda_target})
        col_num_split, row_num_split = T.int32(), T.int32()
        buffer_1_val = T.match_buffer(value_1, (col_num_split, row_num_split))
        col_1_idx = T.match_buffer(c_idx_1, (col_num_split, row_num_split), "int32")
        buffer_1_idx = T.match_buffer(idx_1, (row_num_split,), "int32")
        col_num_real = T.int32()
        vec_val = T.match_buffer(vector_in, (col_num_real,))
        row_num_real = T.int32()
        res_val = T.match_buffer(res_t, (row_num_real,))
        # with T.block("root"):
        if row_num_split > 0:
            with T.block("outer"):
                T.reads(buffer_1_val[0:col_num_split, 0:(row_num_split + 127) // 128 * 128], vec_val[0:col_num_real], col_1_idx[0:col_num_split, 0:(row_num_split + 127) // 128 * 128], res_val[0:row_num_real], buffer_1_idx[0:(row_num_split + 127) // 128 * 128])
                T.writes(res_val[0:row_num_real])
                for real_row_idx_0 in T.thread_binding((row_num_real + 255) // 256, thread="blockIdx.x"):
                    for real_row_idx_1 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("spmv_init"):
                            v_real_row_idx = T.axis.spatial(row_num_real, real_row_idx_0 * 256 + real_row_idx_1)
                            T.where(real_row_idx_0 * 256 + real_row_idx_1 < row_num_real)
                            T.reads()
                            T.writes(res_val[v_real_row_idx])
                            res_val[v_real_row_idx] = T.float32(0.0)
                for row_idx_0 in T.thread_binding((row_num_split + 127) // 128, thread="blockIdx.x"):
                    for row_idx_1 in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("spmv_short_o"):
                            T.reads(buffer_1_val[0:col_num_split, row_idx_0 * 128 + row_idx_1], vec_val[0:col_num_real], col_1_idx[0:col_num_split, row_idx_0 * 128 + row_idx_1], res_val[buffer_1_idx[row_idx_0 * 128 + row_idx_1]], buffer_1_idx[row_idx_0 * 128 + row_idx_1])
                            T.writes()
                            res_local = T.alloc_buffer((1,), scope="local")
                            res_local[0] = T.float32(0.0)
                            if row_idx_0 * 128 + row_idx_1 < row_num_split:
                                for col_idx in range(col_num_split):
                                    with T.block("spmv_short"):
                                        T.reads(res_local[0], buffer_1_val[col_idx, row_idx_0 * 128 + row_idx_1], vec_val[col_1_idx[col_idx, row_idx_0 * 128 + row_idx_1]], col_1_idx[col_idx, row_idx_0 * 128 + row_idx_1])
                                        T.writes(res_local[0])
                                        res_local[0] = res_local[0] + buffer_1_val[col_idx, row_idx_0 * 128 + row_idx_1] * vec_val[col_1_idx[col_idx, row_idx_0 * 128 + row_idx_1]]
                                with T.block("spmv_short_write_back"):
                                    T.reads(res_val[buffer_1_idx[row_idx_0 * 128 + row_idx_1]], buffer_1_idx[row_idx_0 * 128 + row_idx_1], res_local[0])
                                    T.writes()
                                    T.call_extern("float32", "atomicAdd", T.address_of(res_val[buffer_1_idx[row_idx_0 * 128 + row_idx_1]]), res_local[0])
        else:
            with T.block("outer"):
                T.reads()
                T.writes(res_val[0:row_num_real])
                for real_row_idx_0 in T.thread_binding((row_num_real + 255) // 256, thread="blockIdx.x"):
                    for real_row_idx_1 in T.thread_binding(256, thread="threadIdx.x"):
                        with T.block("spmv_init_2"):
                            v_real_row_idx = T.axis.spatial(row_num_real, real_row_idx_0 * 256 + real_row_idx_1)
                            T.where(real_row_idx_0 * 256 + real_row_idx_1 < row_num_real)
                            T.reads()
                            T.writes(res_val[v_real_row_idx])
                            res_val[v_real_row_idx] = T.float32(0.0)
    
    @T.prim_func
    def calculate_buffer_size_cpu(
        indptr: T.handle,
        output: T.handle,
    ) -> None:
        T.func_attr({"target": T.target("llvm"), "func_type": "preprocess"})
        row_add_one = T.int32()
        indptr_buffer = T.match_buffer(indptr, (row_add_one,), "int32")
        output_buffer = T.match_buffer(output, (2,), "int32")
        # [0] : thread_kernel_size
        # [1] : block_kernel_size

        row: T.int32 = row_add_one - 1
        with T.block("root"):
            T.reads(indptr_buffer[0 : row + 1])
            T.writes(output_buffer[0:2])
            for idx in T.serial(2):
                output_buffer[idx] = T.int32(0)
            for row_idx in T.serial(row):
                nnz: T.int32 = indptr_buffer[row_idx + 1] - indptr_buffer[row_idx]
                if nnz <= T.int32(length_threshold):
                    output_buffer[0] += T.floordiv(
                        nnz + T.int32(short_align_val) - 1, T.int32(short_align_val)
                    )
                else:
                    output_buffer[1] += T.floordiv(
                        nnz + T.int32(long_align_val) - 1, T.int32(long_align_val)
                    )
    
    # A：4xint8, slice_id+flag
    @T.prim_func
    def p2p_gpu_process(
        A: T.Buffer((4,), "int8"),
        vector_in: T.handle,
        row_idx_s: T.handle,
        col_idx_s: T.handle,
        val_s: T.handle,
        row_idx_l: T.handle,
        col_idx_l: T.handle,
        val_l: T.handle,
        out: T.handle
    ):
        T.func_attr({"global_symbol": "p2p_gpu_process", "target": cuda_target, "slice_num" : slice_number})
        for b0 in T.thread_binding(1, thread="blockIdx.x"):
                for t0 in T.thread_binding(1, thread="threadIdx.x"):
                    with T.block("wait"):  
                        while  A[1] == 1:
                            # TestP2P.long_lines(val_l, col_idx_l, row_idx_l, vector_in, out)
                            # TestP2P.short_lines_easier(val_s, col_idx_s, row_idx_s, vector_in, out)
                            A[1] == 0
                            
    @R.function(pure=False)
    def foo(indptr: R.Tensor(("origin_row_add_1_",), "int32"), 
            indices: R.Tensor(("origin_nnz_",), "int32"), 
            data: R.Tensor(("origin_nnz_",), "float32"), 
            vector_in: R.Tensor((COL,), "float32"), 
            slice_num: R.Tensor((1,), "int32")) -> R.Tensor(ndim=1, dtype=val_dtype):
        R.func_attr({"global_symbol": "foo"})
        row_s = T.int64()
        row_l = T.int64()

        shape = R.call_tir(
            TestP2P.calculate_buffer_size_cpu,
            (indptr,),
            R.Tensor([2], "int32"),
        )

        splited_num: R.Shape(ndim=2) = R.tensor_to_shape(shape)
        _: R.Shape([row_s, row_l]) = R.match_cast(
            splited_num, R.Shape([row_s, row_l])
        )
        
        indptr_fpga = R.to_vdevice(indptr, "fpga")
        indices_fpga = R.to_vdevice(indices, "fpga")
        data_fpga = R.to_vdevice(data, "fpga")

        out= R.call_tir(TestP2P.fpga_process, (indptr_fpga, indices_fpga, data_fpga), out_sinfo=[
                R.Tensor((4,), "int8"),R.Tensor((row_s,), "int32"),
                R.Tensor((short_align_val, row_s), "int32"),R.Tensor((short_align_val, row_s), "float32"),
                R.Tensor((row_l,), "int32"),R.Tensor((row_l, long_align_val), "int32"),
                R.Tensor((row_l, long_align_val), "float32")]
        ) # 双倍的buffer
        
        flag = R.to_vdevice(R.TupleGetItem(out, 0), "cuda")
        c1 = R.to_vdevice(R.TupleGetItem(out, 1), "cuda")
        c2 = R.to_vdevice(R.TupleGetItem(out, 2), "cuda")
        c3 = R.to_vdevice(R.TupleGetItem(out, 3), "cuda")
        c4 = R.to_vdevice(R.TupleGetItem(out, 4), "cuda")
        c5 = R.to_vdevice(R.TupleGetItem(out, 5), "cuda")
        c6 = R.to_vdevice(R.TupleGetItem(out, 6), "cuda")
        vector = R.to_vdevice(vector_in, "cuda")
        
        # slice_num = R.call_packed("vm.builtin.p2p_descriptor_transfer", slice_num,  sinfo_args=(R.Tensor((1,), "int32")))
		# 一次kernel的launch计算一个slice, kernel名前缀为"p2p_"
        res = R.call_tir(
            TestP2P.p2p_gpu_process, 
            (flag, vector, c1, c2, c3, c4, c5, c6, ),
            out_sinfo=R.Tensor((ROW,), "float32") 
        )

        return res



mod = TestP2P
target = [tvm.cpu(0), tvm.fpga(0), tvm.cuda(0)]
vm = compile(mod, target)
inp = tvm.nd.array(np.random.rand(8, 256, 256).astype(np.float32))
inp1 = tvm.nd.array(np.random.rand(16, 16).astype(np.float32))
inp2 = tvm.nd.array(np.random.rand(1,).astype(np.int32))
res = vm["foo"](inp, inp1, inp2)
tvm.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
# check the resulting tensor is on cpu:0
assert str(res.device) == "fpga(0)"
assert res.device.device_type == 17
assert res.device.device_id == 0