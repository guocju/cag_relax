#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

namespace tvm {
namespace tir {

class MatchBufferRewriter : public StmtExprMutator {
 public:
  explicit MatchBufferRewriter(const PrimFunc& func) : func_(func) {
    // 同步替换 func->params 中被 buffer_map 使用的 Var
    for (const Var& param : func_->params) {
      if (func_->buffer_map.count(param)) {
        // 取出原始 Buffer
        const Buffer& buffer = func_->buffer_map[param];
        // 构造新的 Var 和其对应的 handle 类型
        runtime::DataType dtype = buffer->dtype;
        Type elem_type = PrimType(dtype);
        PointerType handle_type = PointerType(elem_type, "global");
        Var new_var(param->name_hint, handle_type);
        // 记录 Var 重映射
        var_remap_.Set(param, new_var);
        // 构造对应的 Buffer，并绑定到新的 Var
        Buffer new_buf =
            Buffer(new_var, buffer->dtype, buffer->shape, buffer->strides, buffer->elem_offset,
                   buffer->name, buffer->data_alignment, buffer->offset_factor, buffer->buffer_type,
                   buffer->axis_separators, buffer->span);
        // 存储以便后面嵌套 DeclBuffer
        new_buffers_.push_back(new_buf);
        // 更新新的参数列表
        new_params_.push_back(new_var);
      } else {
        // 非 buffer_map 参数保持不变
        new_params_.push_back(param);
      }
    }
  }

  // 重写 Var 引用，进行参数重映射
  PrimExpr VisitExpr_(const VarNode* op) override {
    Var v = GetRef<Var>(op);
    if (var_remap_.count(v)) {
      return var_remap_[v];
    }
    return v;
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    Array<PrimExpr> new_args;
    for (const PrimExpr& param : op->args) {
      bool replaced = false;
      if (const auto var_param = param.as<VarNode>()) {
        String var_name = var_param->name_hint;
        for (const Buffer& buf : new_buffers_) {
          if (buf->name == var_name) {
            new_args.push_back(buf->data);  // 匹配成功，替换为 buffer.data
            replaced = true;
            break;
          }
        }
      }
      if (!replaced) {
        new_args.push_back(param);  // 保留原始参数
      }
    }
    return Call(op->dtype, op->op, new_args, op->span);
  }

  // 执行重写：先替换 Var，再嵌套 DeclBuffer
  PrimFunc Rewrite() {
    // 2. 根据 new_buffers_ 反向嵌套 DeclBuffer
    Stmt nested = func_->body;
    for (int i = static_cast<int>(new_buffers_.size()) - 1; i >= 0; --i) {
      nested = DeclBuffer(new_buffers_[i], nested);
    }
    // 1. 替换函数体内的所有旧 Var 为新 Var
    Stmt remapped = Downcast<Stmt>(StmtMutator::VisitStmt(nested));
    // 3. 构造新的 PrimFunc，清空 buffer_map
    return PrimFunc(new_params_, remapped, func_->ret_type, Map<Var, Buffer>{}, func_->attrs);
  }

 private:
  PrimFunc func_;
  Map<Var, Var> var_remap_;    // 原 Var 到新 Var 的映射
  Array<Buffer> new_buffers_;  // 需要嵌套 DeclBuffer 的 Buffers
  Array<Var> new_params_;      // 重写后的参数列表
};

PrimFunc ReplaceMatchBuffer(const PrimFunc& func) { return MatchBufferRewriter(func).Rewrite(); }

namespace transform {

Pass ReplaceMatchBufferPass() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    IRModule updated_mod = IRModule(mod);

    // 从 module attrs 中获取 entry_func 名称
    Optional<String> entry_func_name = mod->GetAttr<String>("entry_func");

    for (const auto& [gvar, base_func] : mod->functions) {
      // 如果有 entry_func 且当前函数是 entry_func，跳过处理
      if (entry_func_name.defined() && gvar->name_hint == entry_func_name.value()) {
        continue;
      }

      if (auto opt = base_func.as<PrimFunc>()) {
        PrimFunc func = opt.value();
        PrimFunc new_func = ReplaceMatchBuffer(func);
        updated_mod->Add(gvar, new_func);
      }
    }
    return updated_mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.ReplaceMatchBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ReplaceMatchBuffer").set_body_typed(ReplaceMatchBufferPass);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
