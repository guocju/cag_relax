// This pass transforms PrimFunc by extracting hidden shape arguments
// from match_buffer and replacing match_buffer with decl_buffer.

#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

namespace tvm {
namespace tir {

class ExtractBufferShapeRewriter : public StmtMutator {
 public:
  explicit ExtractBufferShapeRewriter(
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>* shape_vars)
      : shape_vars_(shape_vars) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    Array<Buffer> new_alloc_buffers;
    for (const Buffer& buf : op->alloc_buffers) {
      for (PrimExpr dim : buf->shape) {
        if (auto var = dim.as<VarNode>()) {
          shape_vars_->insert(Downcast<Var>(dim));
        }
      }
      new_alloc_buffers.push_back(buf);
    }
    Block new_block =
        Block(op->iter_vars, op->reads, op->writes, op->name_hint, this->VisitStmt(op->body),
              op->init, new_alloc_buffers, op->match_buffers, op->annotations, op->span);
    return std::move(new_block);
  }

 private:
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>* shape_vars_;
};

PrimFunc ExtractBufferShape(PrimFunc func) {
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> shape_vars;
  for (const auto& kv : func->buffer_map) {
    for (PrimExpr dim : kv.second->shape) {
      if (auto var = dim.as<VarNode>()) {
        shape_vars.insert(Downcast<Var>(dim));
      }
    }
  }

  ExtractBufferShapeRewriter rewriter(&shape_vars);
  Stmt new_body = rewriter(func->body);

  Array<Var> new_params = func->params;
  for (const Var& var : shape_vars) {
    bool exists = false;
    for (const Var& param : new_params) {
      if (param.same_as(var)) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      new_params.push_back(var);
    }
  }

  Map<Var, Buffer> new_buffer_map;
  for (const auto& kv : func->buffer_map) {
    new_buffer_map.Set(kv.first, Buffer(kv.second->data, kv.second->dtype, kv.second->shape,
                                        kv.second->strides, kv.second->elem_offset, kv.second->name,
                                        kv.second->data_alignment, kv.second->offset_factor,
                                        kv.second->buffer_type, kv.second->axis_separators));
  }

  return PrimFunc(new_params, new_body, func->ret_type, new_buffer_map, func->attrs);
}

namespace transform {

Pass ExtractBufferShapePass() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    IRModule updated_mod = IRModule(mod);
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        PrimFunc func = opt.value();
        PrimFunc new_func = ExtractBufferShape(func);
        updated_mod->Add(gvar, new_func);
      }
    }
    return updated_mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.ExtractBufferShape", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ExtractBufferShape").set_body_typed(ExtractBufferShapePass);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
