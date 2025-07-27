from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    I.module_attrs({"runtime": None})
    @T.prim_func
    def preprocess_cpu(args: T.handle, arg_type_ids: T.handle("int32"), num_args: T.int32, out_ret_value: T.handle("void"), out_ret_tcode: T.handle("int32"), resource_handle: T.handle) -> T.int32:
        T.func_attr({"calling_conv": 1, "func_type": "preprocess", "target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-pc-linux-gnu", "tag": ""}), "tir.is_entry_func": T.bool(True)})
        stack_tcode: T.handle("int32") = T.tvm_stack_alloca("arg_tcode", 16)
        stack_tcode_1 = T.decl_buffer((T.uint64(16),), "int32", data=stack_tcode)
        stack_value: T.handle = T.tvm_stack_alloca("arg_value", 16)
        assert num_args == 9, "preprocess_cpu: num_args should be 9"
        assert not T.isnullptr(args), "preprocess_cpu: TVMValue* arg pointer was NULL"
        assert not T.isnullptr(arg_type_ids), "preprocess_cpu: int* type_codes was NULL"
        arg_type_ids_1 = T.decl_buffer((9,), "int32", data=arg_type_ids)
        indptr_code: T.int32 = arg_type_ids_1[0]
        assert indptr_code == 3 or indptr_code == 13 or indptr_code == 7 or indptr_code == 4, "preprocess_cpu: Expect arg[0] to be pointer"
        indices_code: T.int32 = arg_type_ids_1[1]
        assert indices_code == 3 or indices_code == 13 or indices_code == 7 or indices_code == 4, "preprocess_cpu: Expect arg[1] to be pointer"
        data_code: T.int32 = arg_type_ids_1[2]
        assert data_code == 3 or data_code == 13 or data_code == 7 or data_code == 4, "preprocess_cpu: Expect arg[2] to be pointer"
        row_idx_s_code: T.int32 = arg_type_ids_1[3]
        assert row_idx_s_code == 3 or row_idx_s_code == 13 or row_idx_s_code == 7 or row_idx_s_code == 4, "preprocess_cpu: Expect arg[3] to be pointer"
        col_idx_s_code: T.int32 = arg_type_ids_1[4]
        assert col_idx_s_code == 3 or col_idx_s_code == 13 or col_idx_s_code == 7 or col_idx_s_code == 4, "preprocess_cpu: Expect arg[4] to be pointer"
        val_s_code: T.int32 = arg_type_ids_1[5]
        assert val_s_code == 3 or val_s_code == 13 or val_s_code == 7 or val_s_code == 4, "preprocess_cpu: Expect arg[5] to be pointer"
        row_idx_l_code: T.int32 = arg_type_ids_1[6]
        assert row_idx_l_code == 3 or row_idx_l_code == 13 or row_idx_l_code == 7 or row_idx_l_code == 4, "preprocess_cpu: Expect arg[6] to be pointer"
        col_idx_l_code: T.int32 = arg_type_ids_1[7]
        assert col_idx_l_code == 3 or col_idx_l_code == 13 or col_idx_l_code == 7 or col_idx_l_code == 4, "preprocess_cpu: Expect arg[7] to be pointer"
        val_l_code: T.int32 = arg_type_ids_1[8]
        assert val_l_code == 3 or val_l_code == 13 or val_l_code == 7 or val_l_code == 4, "preprocess_cpu: Expect arg[8] to be pointer"
        indptr: T.handle = T.tvm_struct_get(args, 0, 12, "handle")
        indices: T.handle = T.tvm_struct_get(args, 1, 12, "handle")
        data: T.handle = T.tvm_struct_get(args, 2, 12, "handle")
        row_idx_s: T.handle = T.tvm_struct_get(args, 3, 12, "handle")
        col_idx_s: T.handle = T.tvm_struct_get(args, 4, 12, "handle")
        val_s: T.handle = T.tvm_struct_get(args, 5, 12, "handle")
        row_idx_l: T.handle = T.tvm_struct_get(args, 6, 12, "handle")
        col_idx_l: T.handle = T.tvm_struct_get(args, 7, 12, "handle")
        val_l: T.handle = T.tvm_struct_get(args, 8, 12, "handle")
        assert not T.isnullptr(indptr), "preprocess_cpu.indptr is expected to have non-NULL DLTensor* pointer"
        assert 1 == T.tvm_struct_get(indptr, 0, 4, "int32"), "preprocess_cpu.indptr.ndim is expected to equal 1"
        preprocess_cpu_indptr_shape: T.handle("int64") = T.tvm_struct_get(indptr, 0, 2, "handle")
        preprocess_cpu_indptr_shape_1 = T.decl_buffer((1,), "int64", data=preprocess_cpu_indptr_shape)
        row_add_one: T.int32 = T.Cast("int32", preprocess_cpu_indptr_shape_1[0])
        preprocess_cpu_indptr_strides: T.handle("int64") = T.tvm_struct_get(indptr, 0, 3, "handle")
        preprocess_cpu_indptr_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_indptr_strides)
        dev_id: T.int32 = T.tvm_struct_get(indptr, 0, 9, "int32")
        indptr_buffer: T.handle("int32", "global") = T.tvm_struct_get(indptr, 0, 1, "handle")
        T.attr(indptr_buffer, "storage_alignment", 64)
        assert not T.isnullptr(indices), "preprocess_cpu.indices is expected to have non-NULL DLTensor* pointer"
        assert 1 == T.tvm_struct_get(indices, 0, 4, "int32"), "preprocess_cpu.indices.ndim is expected to equal 1"
        preprocess_cpu_indices_shape: T.handle("int64") = T.tvm_struct_get(indices, 0, 2, "handle")
        preprocess_cpu_indices_shape_1 = T.decl_buffer((1,), "int64", data=preprocess_cpu_indices_shape)
        nnz4all: T.int32 = T.Cast("int32", preprocess_cpu_indices_shape_1[0])
        preprocess_cpu_indices_strides: T.handle("int64") = T.tvm_struct_get(indices, 0, 3, "handle")
        preprocess_cpu_indices_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_indices_strides)
        indices_buffer: T.handle("int32", "global") = T.tvm_struct_get(indices, 0, 1, "handle")
        T.attr(indices_buffer, "storage_alignment", 64)
        assert not T.isnullptr(data), "preprocess_cpu.data is expected to have non-NULL DLTensor* pointer"
        assert 1 == T.tvm_struct_get(data, 0, 4, "int32"), "preprocess_cpu.data.ndim is expected to equal 1"
        preprocess_cpu_data_shape: T.handle("int64") = T.tvm_struct_get(data, 0, 2, "handle")
        preprocess_cpu_data_shape_1 = T.decl_buffer((1,), "int64", data=preprocess_cpu_data_shape)
        preprocess_cpu_data_strides: T.handle("int64") = T.tvm_struct_get(data, 0, 3, "handle")
        preprocess_cpu_data_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_data_strides)
        data_buffer: T.handle("float32", "global") = T.tvm_struct_get(data, 0, 1, "handle")
        T.attr(data_buffer, "storage_alignment", 64)
        assert not T.isnullptr(row_idx_s), "preprocess_cpu.row_idx_s is expected to have non-NULL DLTensor* pointer"
        assert 1 == T.tvm_struct_get(row_idx_s, 0, 4, "int32"), "preprocess_cpu.row_idx_s.ndim is expected to equal 1"
        preprocess_cpu_row_idx_s_shape: T.handle("int64") = T.tvm_struct_get(row_idx_s, 0, 2, "handle")
        preprocess_cpu_row_idx_s_shape_1 = T.decl_buffer((1,), "int64", data=preprocess_cpu_row_idx_s_shape)
        row_s: T.int64 = preprocess_cpu_row_idx_s_shape_1[0]
        preprocess_cpu_row_idx_s_strides: T.handle("int64") = T.tvm_struct_get(row_idx_s, 0, 3, "handle")
        preprocess_cpu_row_idx_s_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_row_idx_s_strides)
        row_idx_s_buffer: T.handle("int32", "global") = T.tvm_struct_get(row_idx_s, 0, 1, "handle")
        T.attr(row_idx_s_buffer, "storage_alignment", 64)
        assert not T.isnullptr(col_idx_s), "preprocess_cpu.col_idx_s is expected to have non-NULL DLTensor* pointer"
        assert 2 == T.tvm_struct_get(col_idx_s, 0, 4, "int32"), "preprocess_cpu.col_idx_s.ndim is expected to equal 2"
        preprocess_cpu_col_idx_s_shape: T.handle("int64") = T.tvm_struct_get(col_idx_s, 0, 2, "handle")
        preprocess_cpu_col_idx_s_shape_1 = T.decl_buffer((2,), "int64", data=preprocess_cpu_col_idx_s_shape)
        preprocess_cpu_col_idx_s_strides: T.handle("int64") = T.tvm_struct_get(col_idx_s, 0, 3, "handle")
        preprocess_cpu_col_idx_s_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_col_idx_s_strides)
        col_idx_s_buffer: T.handle("int32", "global") = T.tvm_struct_get(col_idx_s, 0, 1, "handle")
        T.attr(col_idx_s_buffer, "storage_alignment", 64)
        assert not T.isnullptr(val_s), "preprocess_cpu.val_s is expected to have non-NULL DLTensor* pointer"
        assert 2 == T.tvm_struct_get(val_s, 0, 4, "int32"), "preprocess_cpu.val_s.ndim is expected to equal 2"
        preprocess_cpu_val_s_shape: T.handle("int64") = T.tvm_struct_get(val_s, 0, 2, "handle")
        preprocess_cpu_val_s_shape_1 = T.decl_buffer((2,), "int64", data=preprocess_cpu_val_s_shape)
        preprocess_cpu_val_s_strides: T.handle("int64") = T.tvm_struct_get(val_s, 0, 3, "handle")
        preprocess_cpu_val_s_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_val_s_strides)
        val_s_buffer: T.handle("float32", "global") = T.tvm_struct_get(val_s, 0, 1, "handle")
        T.attr(val_s_buffer, "storage_alignment", 64)
        assert not T.isnullptr(row_idx_l), "preprocess_cpu.row_idx_l is expected to have non-NULL DLTensor* pointer"
        assert 1 == T.tvm_struct_get(row_idx_l, 0, 4, "int32"), "preprocess_cpu.row_idx_l.ndim is expected to equal 1"
        preprocess_cpu_row_idx_l_shape: T.handle("int64") = T.tvm_struct_get(row_idx_l, 0, 2, "handle")
        preprocess_cpu_row_idx_l_shape_1 = T.decl_buffer((1,), "int64", data=preprocess_cpu_row_idx_l_shape)
        row_l: T.int64 = preprocess_cpu_row_idx_l_shape_1[0]
        preprocess_cpu_row_idx_l_strides: T.handle("int64") = T.tvm_struct_get(row_idx_l, 0, 3, "handle")
        preprocess_cpu_row_idx_l_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_row_idx_l_strides)
        row_idx_l_buffer: T.handle("int32", "global") = T.tvm_struct_get(row_idx_l, 0, 1, "handle")
        T.attr(row_idx_l_buffer, "storage_alignment", 64)
        assert not T.isnullptr(col_idx_l), "preprocess_cpu.col_idx_l is expected to have non-NULL DLTensor* pointer"
        assert 2 == T.tvm_struct_get(col_idx_l, 0, 4, "int32"), "preprocess_cpu.col_idx_l.ndim is expected to equal 2"
        preprocess_cpu_col_idx_l_shape: T.handle("int64") = T.tvm_struct_get(col_idx_l, 0, 2, "handle")
        preprocess_cpu_col_idx_l_shape_1 = T.decl_buffer((2,), "int64", data=preprocess_cpu_col_idx_l_shape)
        preprocess_cpu_col_idx_l_strides: T.handle("int64") = T.tvm_struct_get(col_idx_l, 0, 3, "handle")
        preprocess_cpu_col_idx_l_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_col_idx_l_strides)
        col_idx_l_buffer: T.handle("int32", "global") = T.tvm_struct_get(col_idx_l, 0, 1, "handle")
        T.attr(col_idx_l_buffer, "storage_alignment", 64)
        assert not T.isnullptr(val_l), "preprocess_cpu.val_l is expected to have non-NULL DLTensor* pointer"
        assert 2 == T.tvm_struct_get(val_l, 0, 4, "int32"), "preprocess_cpu.val_l.ndim is expected to equal 2"
        preprocess_cpu_val_l_shape: T.handle("int64") = T.tvm_struct_get(val_l, 0, 2, "handle")
        preprocess_cpu_val_l_shape_1 = T.decl_buffer((2,), "int64", data=preprocess_cpu_val_l_shape)
        preprocess_cpu_val_l_strides: T.handle("int64") = T.tvm_struct_get(val_l, 0, 3, "handle")
        preprocess_cpu_val_l_strides_1 = T.decl_buffer((0,), "int64", data=preprocess_cpu_val_l_strides)
        val_l_buffer: T.handle("float32", "global") = T.tvm_struct_get(val_l, 0, 1, "handle")
        T.attr(val_l_buffer, "storage_alignment", 64)
        assert T.tvm_struct_get(indptr, 0, 5, "uint8") == T.uint8(0) and T.tvm_struct_get(indptr, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(indptr, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.indptr.dtype is expected to be int32"
        if not T.isnullptr(preprocess_cpu_indptr_strides):
            assert row_add_one == 1 or 1 == T.Cast("int32", preprocess_cpu_indptr_strides_1[0]), "preprocess_cpu.indptr.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(indptr, 0, 8, "uint64"), "Argument preprocess_cpu.indptr.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(indptr, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(indptr, 0, 10, "int32") == 17, "Argument preprocess_cpu.indptr.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(indptr, 0, 10, \"int32\")"
        assert row_add_one == 0 or not T.isnullptr(indptr_buffer), "preprocess_cpu.indptr is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(indices, 0, 5, "uint8") == T.uint8(0) and T.tvm_struct_get(indices, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(indices, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.indices.dtype is expected to be int32"
        if not T.isnullptr(preprocess_cpu_indices_strides):
            assert nnz4all == 1 or 1 == T.Cast("int32", preprocess_cpu_indices_strides_1[0]), "preprocess_cpu.indices.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(indices, 0, 8, "uint64"), "Argument preprocess_cpu.indices.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(indices, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(indices, 0, 10, "int32") == 17, "Argument preprocess_cpu.indices.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(indices, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(indices, 0, 9, "int32"), "Argument preprocess_cpu.indices.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(indices, 0, 9, \"int32\")"
        assert nnz4all == 0 or not T.isnullptr(indices_buffer), "preprocess_cpu.indices is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(data, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(data, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(data, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.data.dtype is expected to be float32"
        assert nnz4all == T.Cast("int32", preprocess_cpu_data_shape_1[0]), "Argument preprocess_cpu.data.shape[0] has an unsatisfied constraint: nnz4all == T.Cast(\"int32\", preprocess_cpu_data_shape[0])"
        if not T.isnullptr(preprocess_cpu_data_strides):
            assert nnz4all == 1 or 1 == T.Cast("int32", preprocess_cpu_data_strides_1[0]), "preprocess_cpu.data.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(data, 0, 8, "uint64"), "Argument preprocess_cpu.data.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(data, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(data, 0, 10, "int32") == 17, "Argument preprocess_cpu.data.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(data, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(data, 0, 9, "int32"), "Argument preprocess_cpu.data.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(data, 0, 9, \"int32\")"
        assert nnz4all == 0 or not T.isnullptr(data_buffer), "preprocess_cpu.data is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(row_idx_s, 0, 5, "uint8") == T.uint8(0) and T.tvm_struct_get(row_idx_s, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(row_idx_s, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.row_idx_s.dtype is expected to be int32"
        if not T.isnullptr(preprocess_cpu_row_idx_s_strides):
            assert row_s == T.int64(1) or T.int64(1) == preprocess_cpu_row_idx_s_strides_1[0], "preprocess_cpu.row_idx_s.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(row_idx_s, 0, 8, "uint64"), "Argument preprocess_cpu.row_idx_s.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(row_idx_s, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(row_idx_s, 0, 10, "int32") == 17, "Argument preprocess_cpu.row_idx_s.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(row_idx_s, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(row_idx_s, 0, 9, "int32"), "Argument preprocess_cpu.row_idx_s.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(row_idx_s, 0, 9, \"int32\")"
        assert row_s == T.int64(0) or not T.isnullptr(row_idx_s_buffer), "preprocess_cpu.row_idx_s is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(col_idx_s, 0, 5, "uint8") == T.uint8(0) and T.tvm_struct_get(col_idx_s, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(col_idx_s, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.col_idx_s.dtype is expected to be int32"
        assert T.Cast("int32", preprocess_cpu_col_idx_s_shape_1[0]) == 8, "Argument preprocess_cpu.col_idx_s.shape[0] has an unsatisfied constraint: 8 == T.Cast(\"int32\", preprocess_cpu_col_idx_s_shape[0])"
        assert row_s == preprocess_cpu_col_idx_s_shape_1[1], "Argument preprocess_cpu.col_idx_s.shape[1] has an unsatisfied constraint: row_s == preprocess_cpu_col_idx_s_shape[1]"
        if not T.isnullptr(preprocess_cpu_col_idx_s_strides):
            assert (row_s == T.int64(1) or 1 == T.Cast("int32", preprocess_cpu_col_idx_s_strides_1[1])) and row_s == T.Cast("int64", T.Cast("int32", preprocess_cpu_col_idx_s_strides_1[0])), "preprocess_cpu.col_idx_s.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(col_idx_s, 0, 8, "uint64"), "Argument preprocess_cpu.col_idx_s.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(col_idx_s, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(col_idx_s, 0, 10, "int32") == 17, "Argument preprocess_cpu.col_idx_s.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(col_idx_s, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(col_idx_s, 0, 9, "int32"), "Argument preprocess_cpu.col_idx_s.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(col_idx_s, 0, 9, \"int32\")"
        assert T.int64(8) * row_s == T.int64(0) or not T.isnullptr(col_idx_s_buffer), "preprocess_cpu.col_idx_s is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(val_s, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(val_s, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(val_s, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.val_s.dtype is expected to be float32"
        assert T.Cast("int32", preprocess_cpu_val_s_shape_1[0]) == 8, "Argument preprocess_cpu.val_s.shape[0] has an unsatisfied constraint: 8 == T.Cast(\"int32\", preprocess_cpu_val_s_shape[0])"
        assert row_s == preprocess_cpu_val_s_shape_1[1], "Argument preprocess_cpu.val_s.shape[1] has an unsatisfied constraint: row_s == preprocess_cpu_val_s_shape[1]"
        if not T.isnullptr(preprocess_cpu_val_s_strides):
            assert (row_s == T.int64(1) or 1 == T.Cast("int32", preprocess_cpu_val_s_strides_1[1])) and row_s == T.Cast("int64", T.Cast("int32", preprocess_cpu_val_s_strides_1[0])), "preprocess_cpu.val_s.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(val_s, 0, 8, "uint64"), "Argument preprocess_cpu.val_s.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(val_s, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(val_s, 0, 10, "int32") == 17, "Argument preprocess_cpu.val_s.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(val_s, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(val_s, 0, 9, "int32"), "Argument preprocess_cpu.val_s.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(val_s, 0, 9, \"int32\")"
        assert T.int64(8) * row_s == T.int64(0) or not T.isnullptr(val_s_buffer), "preprocess_cpu.val_s is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(row_idx_l, 0, 5, "uint8") == T.uint8(0) and T.tvm_struct_get(row_idx_l, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(row_idx_l, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.row_idx_l.dtype is expected to be int32"
        if not T.isnullptr(preprocess_cpu_row_idx_l_strides):
            assert row_l == T.int64(1) or T.int64(1) == preprocess_cpu_row_idx_l_strides_1[0], "preprocess_cpu.row_idx_l.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(row_idx_l, 0, 8, "uint64"), "Argument preprocess_cpu.row_idx_l.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(row_idx_l, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(row_idx_l, 0, 10, "int32") == 17, "Argument preprocess_cpu.row_idx_l.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(row_idx_l, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(row_idx_l, 0, 9, "int32"), "Argument preprocess_cpu.row_idx_l.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(row_idx_l, 0, 9, \"int32\")"
        assert row_l == T.int64(0) or not T.isnullptr(row_idx_l_buffer), "preprocess_cpu.row_idx_l is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(col_idx_l, 0, 5, "uint8") == T.uint8(0) and T.tvm_struct_get(col_idx_l, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(col_idx_l, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.col_idx_l.dtype is expected to be int32"
        assert row_l == preprocess_cpu_col_idx_l_shape_1[0], "Argument preprocess_cpu.col_idx_l.shape[0] has an unsatisfied constraint: row_l == preprocess_cpu_col_idx_l_shape[0]"
        assert T.Cast("int32", preprocess_cpu_col_idx_l_shape_1[1]) == 16, "Argument preprocess_cpu.col_idx_l.shape[1] has an unsatisfied constraint: 16 == T.Cast(\"int32\", preprocess_cpu_col_idx_l_shape[1])"
        if not T.isnullptr(preprocess_cpu_col_idx_l_strides):
            assert T.int64(1) == preprocess_cpu_col_idx_l_strides_1[1] and (row_l == T.int64(1) or T.int64(16) == preprocess_cpu_col_idx_l_strides_1[0]), "preprocess_cpu.col_idx_l.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(col_idx_l, 0, 8, "uint64"), "Argument preprocess_cpu.col_idx_l.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(col_idx_l, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(col_idx_l, 0, 10, "int32") == 17, "Argument preprocess_cpu.col_idx_l.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(col_idx_l, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(col_idx_l, 0, 9, "int32"), "Argument preprocess_cpu.col_idx_l.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(col_idx_l, 0, 9, \"int32\")"
        assert row_l * T.int64(16) == T.int64(0) or not T.isnullptr(col_idx_l_buffer), "preprocess_cpu.col_idx_l is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(val_l, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(val_l, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(val_l, 0, 7, "uint16") == T.uint16(1), "preprocess_cpu.val_l.dtype is expected to be float32"
        assert row_l == preprocess_cpu_val_l_shape_1[0], "Argument preprocess_cpu.val_l.shape[0] has an unsatisfied constraint: row_l == preprocess_cpu_val_l_shape[0]"
        assert T.Cast("int32", preprocess_cpu_val_l_shape_1[1]) == 16, "Argument preprocess_cpu.val_l.shape[1] has an unsatisfied constraint: 16 == T.Cast(\"int32\", preprocess_cpu_val_l_shape[1])"
        if not T.isnullptr(preprocess_cpu_val_l_strides):
            assert T.int64(1) == preprocess_cpu_val_l_strides_1[1] and (row_l == T.int64(1) or T.int64(16) == preprocess_cpu_val_l_strides_1[0]), "preprocess_cpu.val_l.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(val_l, 0, 8, "uint64"), "Argument preprocess_cpu.val_l.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(val_l, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(val_l, 0, 10, "int32") == 17, "Argument preprocess_cpu.val_l.device_type has an unsatisfied constraint: 17 == T.tvm_struct_get(val_l, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(val_l, 0, 9, "int32"), "Argument preprocess_cpu.val_l.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(val_l, 0, 9, \"int32\")"
        assert row_l * T.int64(16) == T.int64(0) or not T.isnullptr(val_l_buffer), "preprocess_cpu.val_l is expected to have non-NULL data pointer"
        indptr_buffer_1 = T.decl_buffer((row_add_one,), "int32", data=indptr_buffer)
        indices_buffer_1 = T.decl_buffer((nnz4all,), "int32", data=indices_buffer)
        data_buffer_1 = T.decl_buffer((nnz4all,), data=data_buffer)
        row_idx_s_buffer_1 = T.decl_buffer((row_s,), "int32", data=row_idx_s_buffer)
        col_idx_s_buffer_1 = T.decl_buffer((8, row_s), "int32", data=col_idx_s_buffer)
        val_s_buffer_1 = T.decl_buffer((8, row_s), data=val_s_buffer)
        row_idx_l_buffer_1 = T.decl_buffer((row_l,), "int32", data=row_idx_l_buffer)
        col_idx_l_buffer_1 = T.decl_buffer((row_l, 16), "int32", data=col_idx_l_buffer)
        val_l_buffer_1 = T.decl_buffer((row_l, 16), data=val_l_buffer)
        T.tvm_struct_set(stack_value, 0, 12, T.Cast("int64", 17))
        stack_tcode_1[0] = 0
        T.tvm_struct_set(stack_value, 1, 12, T.Cast("int64", dev_id))
        stack_tcode_1[1] = 0
        T.call_packed_lowered("__tvm_set_device", stack_value, stack_tcode, 0, 2)
        with T.attr(0, "compute_scope", "preprocess_cpu_compute_"):
            status_s: T.handle("int32", "global") = T.TVMBackendAllocWorkspace(17, dev_id, T.uint64(8), 0, 32)
            T.attr(status_s, "storage_alignment", 64)
            if T.isnullptr(status_s):
                T.tvm_throw_last_error()
            with T.LetStmt(T.TVMBackendAllocWorkspace(17, dev_id, T.uint64(8), 0, 32), T.handle("int32", "global")) as status_l:
                T.attr(status_l, "storage_alignment", 64)
                if T.isnullptr(status_l):
                    T.tvm_throw_last_error()
                T.tvm_struct_set(stack_value, 0, 12, col_idx_l_buffer)
                if T.isnullptr(col_idx_l_buffer):
                    stack_tcode_1[0] = 4
                else:
                    stack_tcode_1[0] = 3
                T.tvm_struct_set(stack_value, 1, 12, col_idx_s_buffer)
                if T.isnullptr(col_idx_s_buffer):
                    stack_tcode_1[1] = 4
                else:
                    stack_tcode_1[1] = 3
                T.tvm_struct_set(stack_value, 2, 12, data_buffer)
                if T.isnullptr(data_buffer):
                    stack_tcode_1[2] = 4
                else:
                    stack_tcode_1[2] = 3
                T.tvm_struct_set(stack_value, 3, 12, indices_buffer)
                if T.isnullptr(indices_buffer):
                    stack_tcode_1[3] = 4
                else:
                    stack_tcode_1[3] = 3
                T.tvm_struct_set(stack_value, 4, 12, indptr_buffer)
                if T.isnullptr(indptr_buffer):
                    stack_tcode_1[4] = 4
                else:
                    stack_tcode_1[4] = 3
                T.tvm_struct_set(stack_value, 5, 12, row_idx_l_buffer)
                if T.isnullptr(row_idx_l_buffer):
                    stack_tcode_1[5] = 4
                else:
                    stack_tcode_1[5] = 3
                T.tvm_struct_set(stack_value, 6, 12, row_idx_s_buffer)
                if T.isnullptr(row_idx_s_buffer):
                    stack_tcode_1[6] = 4
                else:
                    stack_tcode_1[6] = 3
                T.tvm_struct_set(stack_value, 7, 12, status_l)
                if T.isnullptr(status_l):
                    stack_tcode_1[7] = 4
                else:
                    stack_tcode_1[7] = 3
                T.tvm_struct_set(stack_value, 8, 12, status_s)
                if T.isnullptr(status_s):
                    stack_tcode_1[8] = 4
                else:
                    stack_tcode_1[8] = 3
                T.tvm_struct_set(stack_value, 9, 12, val_l_buffer)
                if T.isnullptr(val_l_buffer):
                    stack_tcode_1[9] = 4
                else:
                    stack_tcode_1[9] = 3
                T.tvm_struct_set(stack_value, 10, 12, val_s_buffer)
                if T.isnullptr(val_s_buffer):
                    stack_tcode_1[10] = 4
                else:
                    stack_tcode_1[10] = 3
                T.tvm_struct_set(stack_value, 11, 12, T.Cast("int64", nnz4all))
                stack_tcode_1[11] = 0
                T.tvm_struct_set(stack_value, 12, 12, T.Cast("int64", row_add_one))
                stack_tcode_1[12] = 0
                T.tvm_struct_set(stack_value, 13, 12, row_l)
                stack_tcode_1[13] = 0
                T.tvm_struct_set(stack_value, 14, 12, row_s)
                stack_tcode_1[14] = 0
                T.call_packed_lowered("preprocess_cpu_kernel_1", stack_value, stack_tcode, 0, 15)
                if T.TVMBackendFreeWorkspace(17, dev_id, status_l) != 0:
                    T.tvm_throw_last_error()
            if T.TVMBackendFreeWorkspace(17, dev_id, status_s) != 0:
                T.tvm_throw_last_error()
        return 0