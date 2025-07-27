import config
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
import math
import time

val_dtype = "float32"
DEBUG =1
min_nnz_per_slice = 500

def autotune(csr_mtx, dev):
    def get_cost(easier_width_, _, length_threshold_, long_align_val_, b_thread_extent_):
        # (1) get atomic conflict
        row_in_thread_kernel = row_lengths[row_lengths < length_threshold_]
        atomic_conflct_num_thread_kernel = np.sum(np.maximum(((row_in_thread_kernel + easier_width_ - 1) // easier_width_) - 1, 0))  # skip empty rows
        row_in_block_kernel = row_lengths[row_lengths >= length_threshold_]
        atomic_conflct_num_block_kernel = np.sum((row_in_block_kernel + long_align_val_ - 1) // long_align_val_ - 1)
        total_atomic_conflict_num = atomic_conflct_num_thread_kernel + atomic_conflct_num_block_kernel
        # total_atomic_conflct_rate_include_self = (total_atomic_conflict_num) / (ROW_NUM +(total_atomic_conflict_num)) 
        # total_atomic_conflct_rate = (total_atomic_conflict_num) / ROW_NUM

        # TODO: add atomic fix rules
        # row_idx_in_thread_kernel = np.where(row_lengths < length_threshold_)
        # row_idx_in_block_kernel = np.where(row_lengths >= length_threshold_)

        # (2) get padding
        padding_num_thread_kernel = np.sum((easier_width_ - row_in_thread_kernel % easier_width_) % easier_width_)
        # 1. element padding for block kernel rows;
        # 2. costs of shared reduce(transferd to thread kernel padding element);
        # 3. idle thread when block size larger than row length in block kernel;
        padding_num_block_kernel = np.sum(((long_align_val_ - row_in_block_kernel % long_align_val_) % long_align_val_) \
                                        + ((row_in_block_kernel + long_align_val_ - 1) // long_align_val_) * \
                                            (b_thread_extent_ * math.log2(b_thread_extent_) // (t_cycle // s_cycle) \
                                        + max(b_thread_extent_ - long_align_val_, 0)))
        total_padding_num = padding_num_thread_kernel + padding_num_block_kernel
        # total_padding_rate = total_padding_num / NNZ
        # bigger cost means more padding and more atomic conflict also means worse performance
        cost = total_atomic_conflict_num * l2_cycle + total_padding_num * t_cycle
        if DEBUG:
            print("-------------------------------------")
            print(" length threshold:", length_threshold_,
                  " long_align_val:", long_align_val_,
                  " short_align_val:", easier_width_,
                  " long_thread_extent:", b_thread_extent_)
            # print("atomic_conflct_num_thread_kernel:", atomic_conflct_num_thread_kernel)
            # print("atomic_conflct_num_block_kernel:", atomic_conflct_num_block_kernel)
            # print("total_atomic_conflct:", total_atomic_conflict_num)
            # print("total_atomic_conflct_rate_include_self:", total_atomic_conflct_rate_include_self)
            # print("total_atomic_conflct_rate:", total_atomic_conflct_rate)
            # print("padding_num_thread_kernel:", padding_num_thread_kernel)
            # print("padding_num_block_kernel:", padding_num_block_kernel)
            # print("total_padding_num:", total_padding_num)
            # print("total_padding_rate:", total_padding_rate)
            print("cost:", cost)
            print("-------------------------------------")
        return cost

    def get_best_config():
        best_cost = float("inf")
        best_config = None
        threshold_list = [2*i for i in range(7, 13)]
        for length_threshold in threshold_list:
            for b_thread_extent in [128]:
                for easier_width in [2*i for i in range(1, 7)]:
                    for t_thread_extent in [128]:
                        long_align_val = length_threshold // 2
                        cost = get_cost(easier_width, t_thread_extent, length_threshold, long_align_val, b_thread_extent)
                        if cost < best_cost:
                            best_cost = cost
                            best_config = (easier_width, t_thread_extent, length_threshold, long_align_val, b_thread_extent)
        return best_config

    def get_offsets():
        offset_for_slice = [0]
        sum_nnz = 0
        length_threshold = best_config[2]
        short_width = best_config[0]
        long_width = best_config[3]
        for i, val in enumerate(row_lengths):
            k = short_width if val<length_threshold else long_width
            nnz_with_padding = ((val+k-1)//k)*k
            sum_nnz += nnz_with_padding
            if(sum_nnz > min_nnz_per_slice):
                offset_for_slice.append(i+1)
                sum_nnz = 0
        row_nums = len(row_lengths)        
        if offset_for_slice[-1] != row_nums:
            offset_for_slice.append(row_nums)
        return offset_for_slice
            
    print("use analysis to get best config.")
    row_lengths = np.diff(csr_mtx.indptr)
    ROW_NUM = len(row_lengths)
    NNZ = len(csr_mtx.data)
    assert dev in config.device_config_map
    t_cycle = config.device_config_map[dev].thread_consume_element_cycle
    s_cycle = config.device_config_map[dev].shared_consume_element_cycle
    l2_cycle = config.device_config_map[dev].l2_store_latency
    
    best_config = get_best_config()
    offset_for_slice = get_offsets()
    print(offset_for_slice)
    print(f"-------------best config-------------\n thread_kernel_row_length:{best_config[0]},\n" + 
            f" thread_kernel_block_size:{best_config[1]},\n length_threshold:{best_config[2]},\n" +
            f" block_kernel_row_length:{best_config[3]},\n block_kernel_block_size:{best_config[4]}")
    config.update_config(*best_config)
    
def prepare_input():
    matrix_path = "/home/yinuol/code/cag_relax/mytest/data/bcsstk25/bcsstk25.mtx"
    data = mmread(matrix_path)
    original_csr_mat = csr_matrix(data)
    dense_mat = np.abs(original_csr_mat.toarray())
    column_sums = dense_mat.sum(axis=0)
    transition_matrix = dense_mat / column_sums[np.newaxis, :]
    csr_mat = csr_matrix(transition_matrix)

    indptr_np = csr_mat.indptr
    indices_np = csr_mat.indices
    data_np = csr_mat.data.astype(getattr(np, val_dtype))
    COL_NUM = csr_mat.shape[1]
    assert csr_mat.shape[0] == csr_mat.shape[1]
    print("matrix info:\n  name:", matrix_path.split("/")[-1], "\n  shape:", csr_mat.shape, "\n  nnz:", csr_mat.nnz)
    return csr_mat, indptr_np, indices_np, data_np, COL_NUM

if __name__ == "__main__":
    # prepare input
    csr_mat, indptr_np, indices_np, data_np, COL_NUM = prepare_input()
    start_time = time.time()
    autotune(csr_mat, "iluvatar")
    end_time = time.time()
    print(f"autotune 耗时: {end_time - start_time:.4f} 秒")
