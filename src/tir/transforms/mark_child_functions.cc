#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

namespace tvm {
namespace tir {

template <typename T>
using PSet = std::unordered_set<T, ObjectPtrHash, ObjectPtrEqual>;

// Collect all child functions (GlobalVar) invoked anywhere in the module
PSet<GlobalVar> CollectChildFuncs(const IRModule& mod) {
  struct Collector : StmtExprVisitor {
    PSet<GlobalVar>& children;
    Collector(PSet<GlobalVar>& c) : children(c) {}
    void VisitExpr_(const CallNode* call) override {
      if (auto gv = call->op.as<GlobalVar>()) {
        children.insert(gv.value());
      }
      // Recurse into args
      StmtExprVisitor::VisitExpr_(call);
    }
  };

  PSet<GlobalVar> children;
  for (const auto& kv : mod->functions) {
    if (auto pf = kv.second.as<PrimFunc>()) {
      Collector collector(children);
      collector(pf.value()->body);
    }
  }
  return children;
}

namespace transform {

Pass MarkChildFunctions() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    // First, collect all child funcs
    PSet<GlobalVar> children = CollectChildFuncs(mod);
    // Shallow copy to apply updates
    IRModule updated = mod->ShallowCopy();
    // Optionally skip entry function
    Optional<String> entry = mod->GetAttr<String>("entry_func");

    for (const auto& kv : mod->functions) {
      GlobalVar gvar = kv.first;
      if (entry.defined() && gvar->name_hint == entry.value()) {
        continue;
      }
      if (!children.count(gvar)) {
        continue;
      }
      // Only modify PrimFunc
      if (auto pf = kv.second.as<PrimFunc>()) {
        PrimFunc func = pf.value();
        auto opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
        if (opt_target) {
          Target target = opt_target.value();
          if (target->GetHost()) {
            PrimFunc new_func = WithAttr(func, tvm::attr::kTarget, target.WithoutHost());
            func = new_func;
          }
        }
        PrimFunc new_pf = WithAttr(func, tvm::attr::kGlobalSymbol, gvar->name_hint);
        updated->Add(gvar, new_pf, /* override = */ true);
      }
    }
    return updated;
  };

  return tvm::transform::CreateModulePass(pass_func,
                                          /* opt_level = */ 0, "tir.MarkChildFunctions", {});
}

TVM_REGISTER_GLOBAL("tir.transform.MarkChildFunctions").set_body_typed(MarkChildFunctions);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
