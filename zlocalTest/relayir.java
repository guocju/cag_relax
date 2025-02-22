def @main(%onnx::Conv_0 {virtual_device=VirtualDevice(device_type=1, virtual_device_id=0, target=Target(id=59e657a14ad0, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}, host=Target(id=59e657c63830, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"})))}: Tensor[(1, 3, 96, 96), float32] /* ty=Tensor[(1, 3, 96, 96), float32] span=Relu_0.onnx::Conv_0:0:0 */, virtual_device=VirtualDevice(device_type=1, virtual_device_id=0, target=Target(id=59e657a14ad0, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}, host=Target(id=59e657c63830, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"})))) -> Tensor[(1, 2304), float32] {
  %0 = nn.relu(%onnx::Conv_0) /* ty=Tensor[(1, 3, 96, 96), float32] span=Relu_0:0:0 */;
  %1 = on_device(%0, virtual_device=VirtualDevice(device_type=1, virtual_device_id=0, target=Target(id=59e657a14ad0, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}, host=Target(id=59e657c63830, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}))), constrain_result=True) /* ty=Tensor[(1, 3, 96, 96), float32] span=Relu_0:0:0 */;
  %2 = device_copy(%1, src_virtual_device=VirtualDevice(device_type=1, virtual_device_id=0, target=Target(id=59e657a14ad0, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}, host=Target(id=59e657c63830, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}))), dst_virtual_device=VirtualDevice(device_type=17, virtual_device_id=0, target=Target(id=59e657a79020, kind='fpga', host=Target(id=59e657c63830, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"})))) /* ty=Tensor[(1, 3, 96, 96), float32] span=Relu_0:0:0 */;
  %3 = @tvmgen_default_hpu_main_0(%2) /* ty=Tensor[(1, 256, 3, 3), float32] */;
  %4 = on_device(%3, virtual_device=VirtualDevice(device_type=17, virtual_device_id=0, target=Target(id=59e657a79020, kind='fpga', host=Target(id=59e657c63830, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}))), constrain_result=True) /* ty=Tensor[(1, 256, 3, 3), float32] */;
  %5 = device_copy(%4, src_virtual_device=VirtualDevice(device_type=17, virtual_device_id=0, target=Target(id=59e657a79020, kind='fpga', host=Target(id=59e657c63830, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}))), dst_virtual_device=VirtualDevice(device_type=1, virtual_device_id=0, target=Target(id=59e657a14ad0, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"}, host=Target(id=59e657c63830, kind='llvm', keys={'cpu'}, attrs={'mtriple': "x86_64-unknown-linux-gnu"})))) /* ty=Tensor[(1, 256, 3, 3), float32] */;
  %6 = reshape(%5, newshape=[-1, 2304]) /* ty=Tensor[(1, 2304), float32] */;
  nn.softmax(%6, axis=1) /* ty=Tensor[(1, 2304), float32] */
}

def @tvmgen_default_hpu_main_0(%hpu_0_i0: Tensor[(1, 3, 96, 96), float32] /* ty=Tensor[(1, 3, 96, 96), float32] */, Compiler="hpu", Primitive=1, Inline=1, global_symbol="tvmgen_default_hpu_main_0") -> Tensor[(1, 256, 3, 3), float32] {
  %131 = fn (%FunctionVar_22_0: Tensor[(1, 3, 96, 96), float32] /* ty=Tensor[(1, 3, 96, 96), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 32, 96, 96), float32] {
    %126 = nn.conv2d(%FunctionVar_22_0, meta[relay.Constant][66] /* ty=Tensor[(32, 3, 3, 3), float32] span=Conv_3.onnx::Round_613:0:0 */, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 96, 96), float32] span=Conv_3:0:0 */;
    %127 = nn.bias_add(%126, meta[relay.Constant][67] /* ty=Tensor[(32), float32] span=Conv_3.conv1.0.bias:0:0 */) /* ty=Tensor[(1, 32, 96, 96), float32] span=Conv_3:0:0 */;
    %128 = divide(%127, meta[relay.Constant][68] /* ty=Tensor[(1, 32, 1, 1), float32] span=div_input_fakeint32.Conv_3_scale:0:0 */) /* ty=Tensor[(1, 32, 96, 96), float32] span=div_input_fakeint32:0:0 */;
    %129 = floor(%128) /* ty=Tensor[(1, 32, 96, 96), float32] span=floor_input_fakeint32_dived:0:0 */;
    %130 = clip(%129, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 32, 96, 96), float32] span=clip_input_fakeint32_floor:0:0 */;
    nn.relu(%130) /* ty=Tensor[(1, 32, 96, 96), float32] span=Relu_4:0:0 */
  } /* ty=fn (Tensor[(1, 3, 96, 96), float32]) -> Tensor[(1, 32, 96, 96), float32] */;
  %132 = %131(%hpu_0_i0) /* ty=Tensor[(1, 32, 96, 96), float32] */;
  %133 = fn (%FunctionVar_21_0: Tensor[(1, 32, 96, 96), float32] /* ty=Tensor[(1, 32, 96, 96), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 32, 48, 48), float32] {
    %121 = nn.conv2d(%FunctionVar_21_0, meta[relay.Constant][63] /* ty=Tensor[(32, 32, 3, 3), float32] span=Conv_12.onnx::Round_618:0:0 */, strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_12:0:0 */;
    %122 = nn.bias_add(%121, meta[relay.Constant][64] /* ty=Tensor[(32), float32] span=Conv_12.conv1.3.bias:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_12:0:0 */;
    %123 = divide(%122, meta[relay.Constant][65] /* ty=Tensor[(1, 32, 1, 1), float32] span=div_input.3_fakeint32.Conv_12_scale:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=div_input.3_fakeint32:0:0 */;
    %124 = floor(%123) /* ty=Tensor[(1, 32, 48, 48), float32] span=floor_input.3_fakeint32_dived:0:0 */;
    %125 = clip(%124, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 32, 48, 48), float32] span=clip_input.3_fakeint32_floor:0:0 */;
    nn.relu(%125) /* ty=Tensor[(1, 32, 48, 48), float32] span=Relu_13:0:0 */
  } /* ty=fn (Tensor[(1, 32, 96, 96), float32]) -> Tensor[(1, 32, 48, 48), float32] */;
  %134 = %133(%132) /* ty=Tensor[(1, 32, 48, 48), float32] */;
  %135 = fn (%FunctionVar_20_0: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 32, 48, 48), float32] {
    %116 = nn.conv2d(%FunctionVar_20_0, meta[relay.Constant][60] /* ty=Tensor[(32, 32, 3, 3), float32] span=Conv_21.onnx::Round_623:0:0 */, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_21:0:0 */;
    %117 = nn.bias_add(%116, meta[relay.Constant][61] /* ty=Tensor[(32), float32] span=Conv_21.conv2_x.0.residual_function.0.bias:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_21:0:0 */;
    %118 = divide(%117, meta[relay.Constant][62] /* ty=Tensor[(1, 32, 1, 1), float32] span=div_input.7_fakeint32.Conv_21_scale:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=div_input.7_fakeint32:0:0 */;
    %119 = floor(%118) /* ty=Tensor[(1, 32, 48, 48), float32] span=floor_input.7_fakeint32_dived:0:0 */;
    %120 = clip(%119, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 32, 48, 48), float32] span=clip_input.7_fakeint32_floor:0:0 */;
    nn.relu(%120) /* ty=Tensor[(1, 32, 48, 48), float32] span=Relu_22:0:0 */
  } /* ty=fn (Tensor[(1, 32, 48, 48), float32]) -> Tensor[(1, 32, 48, 48), float32] */;
  %136 = %135(%134) /* ty=Tensor[(1, 32, 48, 48), float32] */;
  %137 = fn (%FunctionVar_19_0: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 32, 48, 48), float32] {
    %112 = nn.conv2d(%FunctionVar_19_0, meta[relay.Constant][57] /* ty=Tensor[(32, 32, 3, 3), float32] span=Conv_30.onnx::Round_628:0:0 */, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_30:0:0 */;
    %113 = nn.bias_add(%112, meta[relay.Constant][58] /* ty=Tensor[(32), float32] span=Conv_30.conv2_x.0.residual_function.3.bias:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_30:0:0 */;
    %114 = divide(%113, meta[relay.Constant][59] /* ty=Tensor[(1, 32, 1, 1), float32] span=div_onnx::Div_193_fakeint32.Conv_30_scale:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=div_onnx::Div_193_fakeint32:0:0 */;
    %115 = floor(%114) /* ty=Tensor[(1, 32, 48, 48), float32] span=floor_onnx::Div_193_fakeint32_dived:0:0 */;
    clip(%115, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 32, 48, 48), float32] span=clip_onnx::Div_193_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 32, 48, 48), float32]) -> Tensor[(1, 32, 48, 48), float32] */;
  %138 = %137(%136) /* ty=Tensor[(1, 32, 48, 48), float32] */;
  %139 = fn (%FunctionVar_7_01: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, %FunctionVar_7_1: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, PartitionedFromPattern="add_divide_floor_clip_nn.relu_", Composite="hpu.add_pattern_test_quant") -> Tensor[(1, 32, 48, 48), float32] {
    %108 = add(%FunctionVar_7_01, %FunctionVar_7_1) /* ty=Tensor[(1, 32, 48, 48), float32] span=Add_35:0:0 */;
    %109 = divide(%108, meta[relay.Constant][56] /* ty=Tensor[(1), float32] span=div_input.11_int9.Add_35_scale:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=div_input.11_int9:0:0 */;
    %110 = floor(%109) /* ty=Tensor[(1, 32, 48, 48), float32] span=floor_input.11_int9_dived:0:0 */;
    %111 = clip(%110, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 32, 48, 48), float32] span=clip_input.11_int9_floor:0:0 */;
    nn.relu(%111) /* ty=Tensor[(1, 32, 48, 48), float32] span=Relu_36:0:0 */
  } /* ty=fn (Tensor[(1, 32, 48, 48), float32], Tensor[(1, 32, 48, 48), float32]) -> Tensor[(1, 32, 48, 48), float32] */;
  %140 = %139(%138, %134) /* ty=Tensor[(1, 32, 48, 48), float32] */;
  %141 = fn (%FunctionVar_18_0: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 32, 48, 48), float32] {
    %103 = nn.conv2d(%FunctionVar_18_0, meta[relay.Constant][53] /* ty=Tensor[(32, 32, 3, 3), float32] span=Conv_44.onnx::Round_635:0:0 */, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_44:0:0 */;
    %104 = nn.bias_add(%103, meta[relay.Constant][54] /* ty=Tensor[(32), float32] span=Conv_44.conv2_x.1.residual_function.0.bias:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_44:0:0 */;
    %105 = divide(%104, meta[relay.Constant][55] /* ty=Tensor[(1, 32, 1, 1), float32] span=div_input.15_fakeint32.Conv_44_scale:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=div_input.15_fakeint32:0:0 */;
    %106 = floor(%105) /* ty=Tensor[(1, 32, 48, 48), float32] span=floor_input.15_fakeint32_dived:0:0 */;
    %107 = clip(%106, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 32, 48, 48), float32] span=clip_input.15_fakeint32_floor:0:0 */;
    nn.relu(%107) /* ty=Tensor[(1, 32, 48, 48), float32] span=Relu_45:0:0 */
  } /* ty=fn (Tensor[(1, 32, 48, 48), float32]) -> Tensor[(1, 32, 48, 48), float32] */;
  %142 = %141(%140) /* ty=Tensor[(1, 32, 48, 48), float32] */;
  %143 = fn (%FunctionVar_17_0: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 32, 48, 48), float32] {
    %99 = nn.conv2d(%FunctionVar_17_0, meta[relay.Constant][50] /* ty=Tensor[(32, 32, 3, 3), float32] span=Conv_53.onnx::Round_640:0:0 */, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_53:0:0 */;
    %100 = nn.bias_add(%99, meta[relay.Constant][51] /* ty=Tensor[(32), float32] span=Conv_53.conv2_x.1.residual_function.3.bias:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=Conv_53:0:0 */;
    %101 = divide(%100, meta[relay.Constant][52] /* ty=Tensor[(1, 32, 1, 1), float32] span=div_onnx::Div_238_fakeint32.Conv_53_scale:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=div_onnx::Div_238_fakeint32:0:0 */;
    %102 = floor(%101) /* ty=Tensor[(1, 32, 48, 48), float32] span=floor_onnx::Div_238_fakeint32_dived:0:0 */;
    clip(%102, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 32, 48, 48), float32] span=clip_onnx::Div_238_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 32, 48, 48), float32]) -> Tensor[(1, 32, 48, 48), float32] */;
  %144 = %143(%142) /* ty=Tensor[(1, 32, 48, 48), float32] */;
  %145 = fn (%FunctionVar_6_01: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, %FunctionVar_6_1: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, PartitionedFromPattern="add_divide_floor_clip_nn.relu_", Composite="hpu.add_pattern_test_quant") -> Tensor[(1, 32, 48, 48), float32] {
    %95 = add(%FunctionVar_6_01, %FunctionVar_6_1) /* ty=Tensor[(1, 32, 48, 48), float32] span=Add_58:0:0 */;
    %96 = divide(%95, meta[relay.Constant][49] /* ty=Tensor[(1), float32] span=div_input.19_int9.Add_58_scale:0:0 */) /* ty=Tensor[(1, 32, 48, 48), float32] span=div_input.19_int9:0:0 */;
    %97 = floor(%96) /* ty=Tensor[(1, 32, 48, 48), float32] span=floor_input.19_int9_dived:0:0 */;
    %98 = clip(%97, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 32, 48, 48), float32] span=clip_input.19_int9_floor:0:0 */;
    nn.relu(%98) /* ty=Tensor[(1, 32, 48, 48), float32] span=Relu_59:0:0 */
  } /* ty=fn (Tensor[(1, 32, 48, 48), float32], Tensor[(1, 32, 48, 48), float32]) -> Tensor[(1, 32, 48, 48), float32] */;
  %146 = %145(%144, %140) /* ty=Tensor[(1, 32, 48, 48), float32] */;
  %147 = fn (%FunctionVar_16_0: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 64, 24, 24), float32] {
    %90 = nn.conv2d(%FunctionVar_16_0, meta[relay.Constant][46] /* ty=Tensor[(64, 32, 3, 3), float32] span=Conv_67.onnx::Round_647:0:0 */, strides=[2, 2], padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_67:0:0 */;
    %91 = nn.bias_add(%90, meta[relay.Constant][47] /* ty=Tensor[(64), float32] span=Conv_67.conv3_x.0.residual_function.0.bias:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_67:0:0 */;
    %92 = divide(%91, meta[relay.Constant][48] /* ty=Tensor[(1, 64, 1, 1), float32] span=div_input.23_fakeint32.Conv_67_scale:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=div_input.23_fakeint32:0:0 */;
    %93 = floor(%92) /* ty=Tensor[(1, 64, 24, 24), float32] span=floor_input.23_fakeint32_dived:0:0 */;
    %94 = clip(%93, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 64, 24, 24), float32] span=clip_input.23_fakeint32_floor:0:0 */;
    nn.relu(%94) /* ty=Tensor[(1, 64, 24, 24), float32] span=Relu_68:0:0 */
  } /* ty=fn (Tensor[(1, 32, 48, 48), float32]) -> Tensor[(1, 64, 24, 24), float32] */;
  %148 = %147(%146) /* ty=Tensor[(1, 64, 24, 24), float32] */;
  %149 = fn (%FunctionVar_15_0: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 64, 24, 24), float32] {
    %86 = nn.conv2d(%FunctionVar_15_0, meta[relay.Constant][43] /* ty=Tensor[(64, 64, 3, 3), float32] span=Conv_76.onnx::Round_652:0:0 */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_76:0:0 */;
    %87 = nn.bias_add(%86, meta[relay.Constant][44] /* ty=Tensor[(64), float32] span=Conv_76.conv3_x.0.residual_function.3.bias:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_76:0:0 */;
    %88 = divide(%87, meta[relay.Constant][45] /* ty=Tensor[(1, 64, 1, 1), float32] span=div_onnx::Div_283_fakeint32.Conv_76_scale:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=div_onnx::Div_283_fakeint32:0:0 */;
    %89 = floor(%88) /* ty=Tensor[(1, 64, 24, 24), float32] span=floor_onnx::Div_283_fakeint32_dived:0:0 */;
    clip(%89, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 64, 24, 24), float32] span=clip_onnx::Div_283_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 64, 24, 24), float32]) -> Tensor[(1, 64, 24, 24), float32] */;
  %154 = fn (%FunctionVar_14_0: Tensor[(1, 32, 48, 48), float32] /* ty=Tensor[(1, 32, 48, 48), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 64, 24, 24), float32] {
    %150 = nn.conv2d(%FunctionVar_14_0, meta[relay.Constant][69] /* ty=Tensor[(64, 32, 1, 1), float32] span=Conv_84.onnx::Round_657:0:0 */, strides=[2, 2], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_84:0:0 */;
    %151 = nn.bias_add(%150, meta[relay.Constant][70] /* ty=Tensor[(64), float32] span=Conv_84.conv3_x.0.shortcut.0.bias:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_84:0:0 */;
    %152 = divide(%151, meta[relay.Constant][71] /* ty=Tensor[(1, 64, 1, 1), float32] span=div_onnx::Div_300_fakeint32.Conv_84_scale:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=div_onnx::Div_300_fakeint32:0:0 */;
    %153 = floor(%152) /* ty=Tensor[(1, 64, 24, 24), float32] span=floor_onnx::Div_300_fakeint32_dived:0:0 */;
    clip(%153, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 64, 24, 24), float32] span=clip_onnx::Div_300_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 32, 48, 48), float32]) -> Tensor[(1, 64, 24, 24), float32] */;
  %155 = %149(%148) /* ty=Tensor[(1, 64, 24, 24), float32] */;
  %156 = %154(%146) /* ty=Tensor[(1, 64, 24, 24), float32] */;
  %157 = fn (%FunctionVar_5_01: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, %FunctionVar_5_1: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, PartitionedFromPattern="add_divide_floor_clip_nn.relu_", Composite="hpu.add_pattern_test_quant") -> Tensor[(1, 64, 24, 24), float32] {
    %82 = add(%FunctionVar_5_01, %FunctionVar_5_1) /* ty=Tensor[(1, 64, 24, 24), float32] span=Add_89:0:0 */;
    %83 = divide(%82, meta[relay.Constant][42] /* ty=Tensor[(1), float32] span=div_input.27_int9.Add_89_scale:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=div_input.27_int9:0:0 */;
    %84 = floor(%83) /* ty=Tensor[(1, 64, 24, 24), float32] span=floor_input.27_int9_dived:0:0 */;
    %85 = clip(%84, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 64, 24, 24), float32] span=clip_input.27_int9_floor:0:0 */;
    nn.relu(%85) /* ty=Tensor[(1, 64, 24, 24), float32] span=Relu_90:0:0 */
  } /* ty=fn (Tensor[(1, 64, 24, 24), float32], Tensor[(1, 64, 24, 24), float32]) -> Tensor[(1, 64, 24, 24), float32] */;
  %158 = %157(%155, %156) /* ty=Tensor[(1, 64, 24, 24), float32] */;
  %159 = fn (%FunctionVar_13_0: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 64, 24, 24), float32] {
    %77 = nn.conv2d(%FunctionVar_13_0, meta[relay.Constant][39] /* ty=Tensor[(64, 64, 3, 3), float32] span=Conv_98.onnx::Round_664:0:0 */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_98:0:0 */;
    %78 = nn.bias_add(%77, meta[relay.Constant][40] /* ty=Tensor[(64), float32] span=Conv_98.conv3_x.1.residual_function.0.bias:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_98:0:0 */;
    %79 = divide(%78, meta[relay.Constant][41] /* ty=Tensor[(1, 64, 1, 1), float32] span=div_input.31_fakeint32.Conv_98_scale:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=div_input.31_fakeint32:0:0 */;
    %80 = floor(%79) /* ty=Tensor[(1, 64, 24, 24), float32] span=floor_input.31_fakeint32_dived:0:0 */;
    %81 = clip(%80, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 64, 24, 24), float32] span=clip_input.31_fakeint32_floor:0:0 */;
    nn.relu(%81) /* ty=Tensor[(1, 64, 24, 24), float32] span=Relu_99:0:0 */
  } /* ty=fn (Tensor[(1, 64, 24, 24), float32]) -> Tensor[(1, 64, 24, 24), float32] */;
  %160 = %159(%158) /* ty=Tensor[(1, 64, 24, 24), float32] */;
  %161 = fn (%FunctionVar_12_0: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 64, 24, 24), float32] {
    %73 = nn.conv2d(%FunctionVar_12_0, meta[relay.Constant][36] /* ty=Tensor[(64, 64, 3, 3), float32] span=Conv_107.onnx::Round_669:0:0 */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_107:0:0 */;
    %74 = nn.bias_add(%73, meta[relay.Constant][37] /* ty=Tensor[(64), float32] span=Conv_107.conv3_x.1.residual_function.3.bias:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=Conv_107:0:0 */;
    %75 = divide(%74, meta[relay.Constant][38] /* ty=Tensor[(1, 64, 1, 1), float32] span=div_onnx::Div_345_fakeint32.Conv_107_scale:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=div_onnx::Div_345_fakeint32:0:0 */;
    %76 = floor(%75) /* ty=Tensor[(1, 64, 24, 24), float32] span=floor_onnx::Div_345_fakeint32_dived:0:0 */;
    clip(%76, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 64, 24, 24), float32] span=clip_onnx::Div_345_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 64, 24, 24), float32]) -> Tensor[(1, 64, 24, 24), float32] */;
  %162 = %161(%160) /* ty=Tensor[(1, 64, 24, 24), float32] */;
  %163 = fn (%FunctionVar_4_0: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, %FunctionVar_4_1: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, PartitionedFromPattern="add_divide_floor_clip_nn.relu_", Composite="hpu.add_pattern_test_quant") -> Tensor[(1, 64, 24, 24), float32] {
    %69 = add(%FunctionVar_4_0, %FunctionVar_4_1) /* ty=Tensor[(1, 64, 24, 24), float32] span=Add_112:0:0 */;
    %70 = divide(%69, meta[relay.Constant][35] /* ty=Tensor[(1), float32] span=div_input.35_int9.Add_112_scale:0:0 */) /* ty=Tensor[(1, 64, 24, 24), float32] span=div_input.35_int9:0:0 */;
    %71 = floor(%70) /* ty=Tensor[(1, 64, 24, 24), float32] span=floor_input.35_int9_dived:0:0 */;
    %72 = clip(%71, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 64, 24, 24), float32] span=clip_input.35_int9_floor:0:0 */;
    nn.relu(%72) /* ty=Tensor[(1, 64, 24, 24), float32] span=Relu_113:0:0 */
  } /* ty=fn (Tensor[(1, 64, 24, 24), float32], Tensor[(1, 64, 24, 24), float32]) -> Tensor[(1, 64, 24, 24), float32] */;
  %164 = %163(%162, %158) /* ty=Tensor[(1, 64, 24, 24), float32] */;
  %165 = fn (%FunctionVar_11_0: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 128, 12, 12), float32] {
    %64 = nn.conv2d(%FunctionVar_11_0, meta[relay.Constant][32] /* ty=Tensor[(128, 64, 3, 3), float32] span=Conv_121.onnx::Round_676:0:0 */, strides=[2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_121:0:0 */;
    %65 = nn.bias_add(%64, meta[relay.Constant][33] /* ty=Tensor[(128), float32] span=Conv_121.conv4_x.0.residual_function.0.bias:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_121:0:0 */;
    %66 = divide(%65, meta[relay.Constant][34] /* ty=Tensor[(1, 128, 1, 1), float32] span=div_input.39_fakeint32.Conv_121_scale:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=div_input.39_fakeint32:0:0 */;
    %67 = floor(%66) /* ty=Tensor[(1, 128, 12, 12), float32] span=floor_input.39_fakeint32_dived:0:0 */;
    %68 = clip(%67, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 128, 12, 12), float32] span=clip_input.39_fakeint32_floor:0:0 */;
    nn.relu(%68) /* ty=Tensor[(1, 128, 12, 12), float32] span=Relu_122:0:0 */
  } /* ty=fn (Tensor[(1, 64, 24, 24), float32]) -> Tensor[(1, 128, 12, 12), float32] */;
  %166 = %165(%164) /* ty=Tensor[(1, 128, 12, 12), float32] */;
  %167 = fn (%FunctionVar_10_0: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 128, 12, 12), float32] {
    %60 = nn.conv2d(%FunctionVar_10_0, meta[relay.Constant][29] /* ty=Tensor[(128, 128, 3, 3), float32] span=Conv_130.onnx::Round_681:0:0 */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_130:0:0 */;
    %61 = nn.bias_add(%60, meta[relay.Constant][30] /* ty=Tensor[(128), float32] span=Conv_130.conv4_x.0.residual_function.3.bias:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_130:0:0 */;
    %62 = divide(%61, meta[relay.Constant][31] /* ty=Tensor[(1, 128, 1, 1), float32] span=div_onnx::Div_390_fakeint32.Conv_130_scale:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=div_onnx::Div_390_fakeint32:0:0 */;
    %63 = floor(%62) /* ty=Tensor[(1, 128, 12, 12), float32] span=floor_onnx::Div_390_fakeint32_dived:0:0 */;
    clip(%63, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 128, 12, 12), float32] span=clip_onnx::Div_390_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 128, 12, 12), float32]) -> Tensor[(1, 128, 12, 12), float32] */;
  %172 = fn (%FunctionVar_9_0: Tensor[(1, 64, 24, 24), float32] /* ty=Tensor[(1, 64, 24, 24), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 128, 12, 12), float32] {
    %168 = nn.conv2d(%FunctionVar_9_0, meta[relay.Constant][72] /* ty=Tensor[(128, 64, 1, 1), float32] span=Conv_138.onnx::Round_686:0:0 */, strides=[2, 2], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_138:0:0 */;
    %169 = nn.bias_add(%168, meta[relay.Constant][73] /* ty=Tensor[(128), float32] span=Conv_138.conv4_x.0.shortcut.0.bias:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_138:0:0 */;
    %170 = divide(%169, meta[relay.Constant][74] /* ty=Tensor[(1, 128, 1, 1), float32] span=div_onnx::Div_407_fakeint32.Conv_138_scale:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=div_onnx::Div_407_fakeint32:0:0 */;
    %171 = floor(%170) /* ty=Tensor[(1, 128, 12, 12), float32] span=floor_onnx::Div_407_fakeint32_dived:0:0 */;
    clip(%171, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 128, 12, 12), float32] span=clip_onnx::Div_407_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 64, 24, 24), float32]) -> Tensor[(1, 128, 12, 12), float32] */;
  %173 = %167(%166) /* ty=Tensor[(1, 128, 12, 12), float32] */;
  %174 = %172(%164) /* ty=Tensor[(1, 128, 12, 12), float32] */;
  %175 = fn (%FunctionVar_3_01: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, %FunctionVar_3_1: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, PartitionedFromPattern="add_divide_floor_clip_nn.relu_", Composite="hpu.add_pattern_test_quant") -> Tensor[(1, 128, 12, 12), float32] {
    %56 = add(%FunctionVar_3_01, %FunctionVar_3_1) /* ty=Tensor[(1, 128, 12, 12), float32] span=Add_143:0:0 */;
    %57 = divide(%56, meta[relay.Constant][28] /* ty=Tensor[(1), float32] span=div_input.43_int9.Add_143_scale:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=div_input.43_int9:0:0 */;
    %58 = floor(%57) /* ty=Tensor[(1, 128, 12, 12), float32] span=floor_input.43_int9_dived:0:0 */;
    %59 = clip(%58, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 128, 12, 12), float32] span=clip_input.43_int9_floor:0:0 */;
    nn.relu(%59) /* ty=Tensor[(1, 128, 12, 12), float32] span=Relu_144:0:0 */
  } /* ty=fn (Tensor[(1, 128, 12, 12), float32], Tensor[(1, 128, 12, 12), float32]) -> Tensor[(1, 128, 12, 12), float32] */;
  %176 = %175(%173, %174) /* ty=Tensor[(1, 128, 12, 12), float32] */;
  %177 = fn (%FunctionVar_8_0: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 128, 12, 12), float32] {
    %51 = nn.conv2d(%FunctionVar_8_0, meta[relay.Constant][25] /* ty=Tensor[(128, 128, 3, 3), float32] span=Conv_152.onnx::Round_693:0:0 */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_152:0:0 */;
    %52 = nn.bias_add(%51, meta[relay.Constant][26] /* ty=Tensor[(128), float32] span=Conv_152.conv4_x.1.residual_function.0.bias:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_152:0:0 */;
    %53 = divide(%52, meta[relay.Constant][27] /* ty=Tensor[(1, 128, 1, 1), float32] span=div_input.47_fakeint32.Conv_152_scale:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=div_input.47_fakeint32:0:0 */;
    %54 = floor(%53) /* ty=Tensor[(1, 128, 12, 12), float32] span=floor_input.47_fakeint32_dived:0:0 */;
    %55 = clip(%54, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 128, 12, 12), float32] span=clip_input.47_fakeint32_floor:0:0 */;
    nn.relu(%55) /* ty=Tensor[(1, 128, 12, 12), float32] span=Relu_153:0:0 */
  } /* ty=fn (Tensor[(1, 128, 12, 12), float32]) -> Tensor[(1, 128, 12, 12), float32] */;
  %178 = %177(%176) /* ty=Tensor[(1, 128, 12, 12), float32] */;
  %179 = fn (%FunctionVar_7_0: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 128, 12, 12), float32] {
    %47 = nn.conv2d(%FunctionVar_7_0, meta[relay.Constant][22] /* ty=Tensor[(128, 128, 3, 3), float32] span=Conv_161.onnx::Round_698:0:0 */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_161:0:0 */;
    %48 = nn.bias_add(%47, meta[relay.Constant][23] /* ty=Tensor[(128), float32] span=Conv_161.conv4_x.1.residual_function.3.bias:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=Conv_161:0:0 */;
    %49 = divide(%48, meta[relay.Constant][24] /* ty=Tensor[(1, 128, 1, 1), float32] span=div_onnx::Div_452_fakeint32.Conv_161_scale:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=div_onnx::Div_452_fakeint32:0:0 */;
    %50 = floor(%49) /* ty=Tensor[(1, 128, 12, 12), float32] span=floor_onnx::Div_452_fakeint32_dived:0:0 */;
    clip(%50, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 128, 12, 12), float32] span=clip_onnx::Div_452_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 128, 12, 12), float32]) -> Tensor[(1, 128, 12, 12), float32] */;
  %180 = %179(%178) /* ty=Tensor[(1, 128, 12, 12), float32] */;
  %181 = fn (%FunctionVar_2_01: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, %FunctionVar_2_1: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, PartitionedFromPattern="add_divide_floor_clip_nn.relu_", Composite="hpu.add_pattern_test_quant") -> Tensor[(1, 128, 12, 12), float32] {
    %43 = add(%FunctionVar_2_01, %FunctionVar_2_1) /* ty=Tensor[(1, 128, 12, 12), float32] span=Add_166:0:0 */;
    %44 = divide(%43, meta[relay.Constant][21] /* ty=Tensor[(1), float32] span=div_input.51_int9.Add_166_scale:0:0 */) /* ty=Tensor[(1, 128, 12, 12), float32] span=div_input.51_int9:0:0 */;
    %45 = floor(%44) /* ty=Tensor[(1, 128, 12, 12), float32] span=floor_input.51_int9_dived:0:0 */;
    %46 = clip(%45, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 128, 12, 12), float32] span=clip_input.51_int9_floor:0:0 */;
    nn.relu(%46) /* ty=Tensor[(1, 128, 12, 12), float32] span=Relu_167:0:0 */
  } /* ty=fn (Tensor[(1, 128, 12, 12), float32], Tensor[(1, 128, 12, 12), float32]) -> Tensor[(1, 128, 12, 12), float32] */;
  %182 = %181(%180, %176) /* ty=Tensor[(1, 128, 12, 12), float32] */;
  %183 = fn (%FunctionVar_6_0: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 256, 6, 6), float32] {
    %38 = nn.conv2d(%FunctionVar_6_0, meta[relay.Constant][18] /* ty=Tensor[(256, 128, 3, 3), float32] span=Conv_175.onnx::Round_705:0:0 */, strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_175:0:0 */;
    %39 = nn.bias_add(%38, meta[relay.Constant][19] /* ty=Tensor[(256), float32] span=Conv_175.conv5_x.0.residual_function.0.bias:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_175:0:0 */;
    %40 = divide(%39, meta[relay.Constant][20] /* ty=Tensor[(1, 256, 1, 1), float32] span=div_input.55_fakeint32.Conv_175_scale:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=div_input.55_fakeint32:0:0 */;
    %41 = floor(%40) /* ty=Tensor[(1, 256, 6, 6), float32] span=floor_input.55_fakeint32_dived:0:0 */;
    %42 = clip(%41, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 256, 6, 6), float32] span=clip_input.55_fakeint32_floor:0:0 */;
    nn.relu(%42) /* ty=Tensor[(1, 256, 6, 6), float32] span=Relu_176:0:0 */
  } /* ty=fn (Tensor[(1, 128, 12, 12), float32]) -> Tensor[(1, 256, 6, 6), float32] */;
  %184 = %183(%182) /* ty=Tensor[(1, 256, 6, 6), float32] */;
  %185 = fn (%FunctionVar_5_0: Tensor[(1, 256, 6, 6), float32] /* ty=Tensor[(1, 256, 6, 6), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 256, 6, 6), float32] {
    %34 = nn.conv2d(%FunctionVar_5_0, meta[relay.Constant][15] /* ty=Tensor[(256, 256, 3, 3), float32] span=Conv_184.onnx::Round_710:0:0 */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_184:0:0 */;
    %35 = nn.bias_add(%34, meta[relay.Constant][16] /* ty=Tensor[(256), float32] span=Conv_184.conv5_x.0.residual_function.3.bias:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_184:0:0 */;
    %36 = divide(%35, meta[relay.Constant][17] /* ty=Tensor[(1, 256, 1, 1), float32] span=div_onnx::Div_497_fakeint32.Conv_184_scale:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=div_onnx::Div_497_fakeint32:0:0 */;
    %37 = floor(%36) /* ty=Tensor[(1, 256, 6, 6), float32] span=floor_onnx::Div_497_fakeint32_dived:0:0 */;
    clip(%37, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 256, 6, 6), float32] span=clip_onnx::Div_497_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 256, 6, 6), float32]) -> Tensor[(1, 256, 6, 6), float32] */;
  %190 = fn (%FunctionVar_4_01: Tensor[(1, 128, 12, 12), float32] /* ty=Tensor[(1, 128, 12, 12), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 256, 6, 6), float32] {
    %186 = nn.conv2d(%FunctionVar_4_01, meta[relay.Constant][75] /* ty=Tensor[(256, 128, 1, 1), float32] span=Conv_192.onnx::Round_715:0:0 */, strides=[2, 2], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_192:0:0 */;
    %187 = nn.bias_add(%186, meta[relay.Constant][76] /* ty=Tensor[(256), float32] span=Conv_192.conv5_x.0.shortcut.0.bias:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_192:0:0 */;
    %188 = divide(%187, meta[relay.Constant][77] /* ty=Tensor[(1, 256, 1, 1), float32] span=div_onnx::Div_514_fakeint32.Conv_192_scale:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=div_onnx::Div_514_fakeint32:0:0 */;
    %189 = floor(%188) /* ty=Tensor[(1, 256, 6, 6), float32] span=floor_onnx::Div_514_fakeint32_dived:0:0 */;
    clip(%189, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 256, 6, 6), float32] span=clip_onnx::Div_514_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 128, 12, 12), float32]) -> Tensor[(1, 256, 6, 6), float32] */;
  %191 = %185(%184) /* ty=Tensor[(1, 256, 6, 6), float32] */;
  %192 = %190(%182) /* ty=Tensor[(1, 256, 6, 6), float32] */;
  %193 = fn (%FunctionVar_1_01: Tensor[(1, 256, 6, 6), float32] /* ty=Tensor[(1, 256, 6, 6), float32] */, %FunctionVar_1_1: Tensor[(1, 256, 6, 6), float32] /* ty=Tensor[(1, 256, 6, 6), float32] */, PartitionedFromPattern="add_divide_floor_clip_nn.relu_", Composite="hpu.add_pattern_test_quant") -> Tensor[(1, 256, 6, 6), float32] {
    %30 = add(%FunctionVar_1_01, %FunctionVar_1_1) /* ty=Tensor[(1, 256, 6, 6), float32] span=Add_197:0:0 */;
    %31 = divide(%30, meta[relay.Constant][14] /* ty=Tensor[(1), float32] span=div_input.59_int9.Add_197_scale:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=div_input.59_int9:0:0 */;
    %32 = floor(%31) /* ty=Tensor[(1, 256, 6, 6), float32] span=floor_input.59_int9_dived:0:0 */;
    %33 = clip(%32, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 256, 6, 6), float32] span=clip_input.59_int9_floor:0:0 */;
    nn.relu(%33) /* ty=Tensor[(1, 256, 6, 6), float32] span=Relu_198:0:0 */
  } /* ty=fn (Tensor[(1, 256, 6, 6), float32], Tensor[(1, 256, 6, 6), float32]) -> Tensor[(1, 256, 6, 6), float32] */;
  %194 = %193(%191, %192) /* ty=Tensor[(1, 256, 6, 6), float32] */;
  %195 = fn (%FunctionVar_3_0: Tensor[(1, 256, 6, 6), float32] /* ty=Tensor[(1, 256, 6, 6), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 256, 6, 6), float32] {
    %25 = nn.conv2d(%FunctionVar_3_0, meta[relay.Constant][11] /* ty=Tensor[(256, 256, 3, 3), float32] span=Conv_206.onnx::Round_722:0:0 */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_206:0:0 */;
    %26 = nn.bias_add(%25, meta[relay.Constant][12] /* ty=Tensor[(256), float32] span=Conv_206.conv5_x.1.residual_function.0.bias:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_206:0:0 */;
    %27 = divide(%26, meta[relay.Constant][13] /* ty=Tensor[(1, 256, 1, 1), float32] span=div_input.63_fakeint32.Conv_206_scale:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=div_input.63_fakeint32:0:0 */;
    %28 = floor(%27) /* ty=Tensor[(1, 256, 6, 6), float32] span=floor_input.63_fakeint32_dived:0:0 */;
    %29 = clip(%28, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 256, 6, 6), float32] span=clip_input.63_fakeint32_floor:0:0 */;
    nn.relu(%29) /* ty=Tensor[(1, 256, 6, 6), float32] span=Relu_207:0:0 */
  } /* ty=fn (Tensor[(1, 256, 6, 6), float32]) -> Tensor[(1, 256, 6, 6), float32] */;
  %196 = %195(%194) /* ty=Tensor[(1, 256, 6, 6), float32] */;
  %197 = fn (%FunctionVar_2_0: Tensor[(1, 256, 6, 6), float32] /* ty=Tensor[(1, 256, 6, 6), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_", Composite="hpu.base_conv_quant") -> Tensor[(1, 256, 6, 6), float32] {
    %21 = nn.conv2d(%FunctionVar_2_0, meta[relay.Constant][8] /* ty=Tensor[(256, 256, 3, 3), float32] span=Conv_215.onnx::Round_727:0:0 */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_215:0:0 */;
    %22 = nn.bias_add(%21, meta[relay.Constant][9] /* ty=Tensor[(256), float32] span=Conv_215.conv5_x.1.residual_function.3.bias:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=Conv_215:0:0 */;
    %23 = divide(%22, meta[relay.Constant][10] /* ty=Tensor[(1, 256, 1, 1), float32] span=div_onnx::Div_559_fakeint32.Conv_215_scale:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=div_onnx::Div_559_fakeint32:0:0 */;
    %24 = floor(%23) /* ty=Tensor[(1, 256, 6, 6), float32] span=floor_onnx::Div_559_fakeint32_dived:0:0 */;
    clip(%24, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 256, 6, 6), float32] span=clip_onnx::Div_559_fakeint32_floor:0:0 */
  } /* ty=fn (Tensor[(1, 256, 6, 6), float32]) -> Tensor[(1, 256, 6, 6), float32] */;
  %198 = %197(%196) /* ty=Tensor[(1, 256, 6, 6), float32] */;
  %199 = fn (%FunctionVar_0_01: Tensor[(1, 256, 6, 6), float32] /* ty=Tensor[(1, 256, 6, 6), float32] */, %FunctionVar_0_1: Tensor[(1, 256, 6, 6), float32] /* ty=Tensor[(1, 256, 6, 6), float32] */, PartitionedFromPattern="add_divide_floor_clip_nn.relu_", Composite="hpu.add_pattern_test_quant") -> Tensor[(1, 256, 6, 6), float32] {
    %17 = add(%FunctionVar_0_01, %FunctionVar_0_1) /* ty=Tensor[(1, 256, 6, 6), float32] span=Add_220:0:0 */;
    %18 = divide(%17, meta[relay.Constant][7] /* ty=Tensor[(1), float32] span=div_input.67_int9.Add_220_scale:0:0 */) /* ty=Tensor[(1, 256, 6, 6), float32] span=div_input.67_int9:0:0 */;
    %19 = floor(%18) /* ty=Tensor[(1, 256, 6, 6), float32] span=floor_input.67_int9_dived:0:0 */;
    %20 = clip(%19, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 256, 6, 6), float32] span=clip_input.67_int9_floor:0:0 */;
    nn.relu(%20) /* ty=Tensor[(1, 256, 6, 6), float32] span=Relu_221:0:0 */
  } /* ty=fn (Tensor[(1, 256, 6, 6), float32], Tensor[(1, 256, 6, 6), float32]) -> Tensor[(1, 256, 6, 6), float32] */;
  %200 = %199(%198, %194) /* ty=Tensor[(1, 256, 6, 6), float32] */;
  %201 = fn (%FunctionVar_1_0: Tensor[(1, 256, 6, 6), float32] /* ty=Tensor[(1, 256, 6, 6), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_nn.relu_", Composite="hpu.base_conv_quant") -> Tensor[(1, 512, 3, 3), float32] {
    %12 = nn.conv2d(%FunctionVar_1_0, meta[relay.Constant][4] /* ty=Tensor[(512, 256, 3, 3), float32] span=Conv_229.onnx::Round_734:0:0 */, strides=[2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 3, 3), float32] span=Conv_229:0:0 */;
    %13 = nn.bias_add(%12, meta[relay.Constant][5] /* ty=Tensor[(512), float32] span=Conv_229.conv6.0.bias:0:0 */) /* ty=Tensor[(1, 512, 3, 3), float32] span=Conv_229:0:0 */;
    %14 = divide(%13, meta[relay.Constant][6] /* ty=Tensor[(1, 512, 1, 1), float32] span=div_input.71_fakeint32.Conv_229_scale:0:0 */) /* ty=Tensor[(1, 512, 3, 3), float32] span=div_input.71_fakeint32:0:0 */;
    %15 = floor(%14) /* ty=Tensor[(1, 512, 3, 3), float32] span=floor_input.71_fakeint32_dived:0:0 */;
    %16 = clip(%15, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 512, 3, 3), float32] span=clip_input.71_fakeint32_floor:0:0 */;
    nn.relu(%16) /* ty=Tensor[(1, 512, 3, 3), float32] span=Relu_230:0:0 */
  } /* ty=fn (Tensor[(1, 256, 6, 6), float32]) -> Tensor[(1, 512, 3, 3), float32] */;
  %202 = %201(%200) /* ty=Tensor[(1, 512, 3, 3), float32] */;
  %203 = fn (%FunctionVar_0_0: Tensor[(1, 512, 3, 3), float32] /* ty=Tensor[(1, 512, 3, 3), float32] */, PartitionedFromPattern="nn.conv2d_nn.bias_add_divide_floor_clip_divide_", Composite="hpu.base_conv_quant") -> Tensor[(1, 256, 3, 3), float32] {
    %7 = nn.conv2d(%FunctionVar_0_0, meta[relay.Constant][0] /* ty=Tensor[(256, 512, 3, 3), float32] span=Conv_238.onnx::Round_739:0:0 */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 3, 3), float32] span=Conv_238:0:0 */;
    %8 = nn.bias_add(%7, meta[relay.Constant][1] /* ty=Tensor[(256), float32] span=Conv_238.conv7.0.bias:0:0 */) /* ty=Tensor[(1, 256, 3, 3), float32] span=Conv_238:0:0 */;
    %9 = divide(%8, meta[relay.Constant][2] /* ty=Tensor[(1, 256, 1, 1), float32] span=div_onnx::Div_604_fakeint32.Conv_238_scale:0:0 */) /* ty=Tensor[(1, 256, 3, 3), float32] span=div_onnx::Div_604_fakeint32:0:0 */;
    %10 = floor(%9) /* ty=Tensor[(1, 256, 3, 3), float32] span=floor_onnx::Div_604_fakeint32_dived:0:0 */;
    %11 = clip(%10, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 256, 3, 3), float32] span=clip_onnx::Div_604_fakeint32_floor:0:0 */;
    divide(%11, meta[relay.Constant][3] /* ty=Tensor[(1), float32] span=div_onnx::Div_604_int8.dummy_conv7.a_quantizer.scale_output_scale:0:0 */) /* ty=Tensor[(1, 256, 3, 3), float32] span=div_onnx::Div_604_int8:0:0 */
  } /* ty=fn (Tensor[(1, 512, 3, 3), float32]) -> Tensor[(1, 256, 3, 3), float32] */;
  %203(%202) /* ty=Tensor[(1, 256, 3, 3), float32] */
}

