#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class ThreadBindingChecker : public StmtVisitor {
 public:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      // These attributes are only allowed in device-side code, so
      // they should be annotated with the function's default target.
      has_thread_binding_ = true;
    } else {
      // All other annotations are ignored
      StmtVisitor::VisitStmt_(op);
    }
  }

  bool has_thread_binding_{false};
};

PrimFunc WrapPrimFuncWithThreadBinding(PrimFunc f) {
  // 检查是否有 thread_binding
  ThreadBindingChecker checker;
  checker(f->body);
  if (checker.has_thread_binding_) {
    return f;  // 已有绑定，返回原始函数
  }

  Var thread_idx_var("threadIdx_x", DataType::Int(32));
  Range dom = Range::FromMinExtent(0, 1);
  IterVar thread_iter = IterVar(dom, thread_idx_var, IterVarType::kThreadIndex, "threadIdx.x");

  // 构造绑定线程的 For 循环
  Stmt wrapped_body = For(
      /*loop_var=*/thread_idx_var,
      /*min=*/0,
      /*extent=*/1,
      /*kind=*/ForKind::kThreadBinding,
      /*body=*/f->body,
      /*thread_binding=*/thread_iter,
      /*annotations=*/{},
      /*span=*/Span());

  // 返回新的 PrimFunc
  return PrimFunc(f->params, wrapped_body, f->ret_type, f->buffer_map, f->attrs);
}

tvm::transform::Pass WrapWithThreadBindingPass() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    IRModule updated_mod = mod->ShallowCopy();
    Optional<String> entry_func_name = mod->GetAttr<String>("entry_func");

    for (const auto& [gvar, base_func] : mod->functions) {
      if (entry_func_name.defined() && gvar->name_hint == entry_func_name.value()) {
        continue;
      }
      if (auto opt = base_func.as<PrimFunc>()) {
        PrimFunc func = opt.value();
        PrimFunc new_func = WrapPrimFuncWithThreadBinding(func);
        updated_mod->Add(gvar, new_func);
      }
    }

    return updated_mod;
  };

  return tvm::transform::CreateModulePass(pass_func,
                                          /*opt_level=*/0, "tir.AddThreadBinding", {});
}

TVM_REGISTER_GLOBAL("tir.transform.AddThreadBinding").set_body_typed(WrapWithThreadBindingPass);

}  // namespace tir
}  // namespace tvm
