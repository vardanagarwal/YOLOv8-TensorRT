import os
import pickle
from collections import defaultdict, namedtuple, OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np


import onnx
import tensorrt as trt
import torch

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


class EngineBuilder:
    seg = False

    def __init__(
            self,
            checkpoint: Union[str, Path],
            device: Optional[Union[str, int, torch.device]] = None) -> None:
        checkpoint = Path(checkpoint) if isinstance(checkpoint,
                                                    str) else checkpoint
        assert checkpoint.exists() and checkpoint.suffix in ('.onnx', '.pkl')
        self.api = checkpoint.suffix == '.pkl'
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')

        self.checkpoint = checkpoint
        self.device = device

    def __build_engine(self,
                       fp16: bool = True,
                       input_shape: Union[List, Tuple] = (1, 3, 640, 640),
                       iou_thres: float = 0.65,
                       conf_thres: float = 0.25,
                       topk: int = 100,
                       with_profiling: bool = True) -> None:
        logger = trt.Logger(trt.Logger.INFO)
        logger.min_severity = trt.Logger.Severity.VERBOSE
        trt.init_libnvinfer_plugins(logger, namespace='')
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        # config.max_workspace_size = torch.cuda.get_device_properties(
        #     self.device).total_memory
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10 
        # workspace = int(self.args.workspace * (1 << 30)) if self.args.workspace is not None else 0
        workspace = 0
        if is_trt10 and workspace > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        elif workspace > 0:  # TensorRT versions 7, 8
            config.max_workspace_size = workspace
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)

        self.logger = logger
        self.builder = builder
        self.network = network
        if self.api:
            self.build_from_api(fp16, input_shape, iou_thres, conf_thres, topk)
        else:
            self.build_from_onnx(iou_thres, conf_thres, topk)
        if fp16 and self.builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        self.weight = self.checkpoint.with_suffix('.engine')

        if with_profiling:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        with self.builder.build_serialized_network(self.network, config)  if is_trt10 else self.builder.build_engine(self.network, config) as engine:
            # self.weight.write_bytes(engine.serialize())
            self.weight.write_bytes(engine if is_trt10 else engine.serialize())
        self.logger.log(
            trt.Logger.WARNING, f'Build tensorrt engine finish.\n'
            f'Save in {str(self.weight.absolute())}')

    def build(self,
              fp16: bool = True,
              input_shape: Union[List, Tuple] = (1, 3, 640, 640),
              iou_thres: float = 0.65,
              conf_thres: float = 0.25,
              topk: int = 100,
              with_profiling=True) -> None:
        self.__build_engine(fp16, input_shape, iou_thres, conf_thres, topk,
                            with_profiling)

    def build_from_onnx(self,
                        iou_thres: float = 0.65,
                        conf_thres: float = 0.25,
                        topk: int = 100):
        parser = trt.OnnxParser(self.network, self.logger)
        onnx_model = onnx.load(str(self.checkpoint))
        if not self.seg:
            onnx_model.graph.node[-1].attribute[2].i = topk
            onnx_model.graph.node[-1].attribute[3].f = conf_thres
            onnx_model.graph.node[-1].attribute[4].f = iou_thres

        if not parser.parse(onnx_model.SerializeToString()):
            raise RuntimeError(
                f'failed to load ONNX file: {str(self.checkpoint)}')
        inputs = [
            self.network.get_input(i) for i in range(self.network.num_inputs)
        ]
        outputs = [
            self.network.get_output(i) for i in range(self.network.num_outputs)
        ]

        for inp in inputs:
            self.logger.log(
                trt.Logger.WARNING,
                f'input "{inp.name}" with shape: {inp.shape} '
                f'dtype: {inp.dtype}')
        for out in outputs:
            self.logger.log(
                trt.Logger.WARNING,
                f'output "{out.name}" with shape: {out.shape} '
                f'dtype: {out.dtype}')

    def build_from_api(
        self,
        fp16: bool = True,
        input_shape: Union[List, Tuple] = (1, 3, 640, 640),
        iou_thres: float = 0.65,
        conf_thres: float = 0.25,
        topk: int = 100,
    ):
        assert not self.seg
        from .api import SPPF, C2f, Conv, Detect, get_depth, get_width

        with open(self.checkpoint, 'rb') as f:
            state_dict = pickle.load(f)
        mapping = {0.25: 1024, 0.5: 1024, 0.75: 768, 1.0: 512, 1.25: 512}

        GW = state_dict['GW']
        GD = state_dict['GD']
        width_64 = get_width(64, GW)
        width_128 = get_width(128, GW)
        width_256 = get_width(256, GW)
        width_512 = get_width(512, GW)
        width_1024 = get_width(mapping[GW], GW)
        depth_3 = get_depth(3, GD)
        depth_6 = get_depth(6, GD)
        strides = state_dict['strides']
        reg_max = state_dict['reg_max']
        images = self.network.add_input(name='images',
                                        dtype=trt.float32,
                                        shape=trt.Dims4(input_shape))
        assert images, 'Add input failed'

        Conv_0 = Conv(self.network, state_dict, images, width_64, 3, 2, 1,
                      'Conv.0')
        Conv_1 = Conv(self.network, state_dict, Conv_0.get_output(0),
                      width_128, 3, 2, 1, 'Conv.1')
        C2f_2 = C2f(self.network, state_dict, Conv_1.get_output(0), width_128,
                    depth_3, True, 1, 0.5, 'C2f.2')
        Conv_3 = Conv(self.network, state_dict, C2f_2.get_output(0), width_256,
                      3, 2, 1, 'Conv.3')
        C2f_4 = C2f(self.network, state_dict, Conv_3.get_output(0), width_256,
                    depth_6, True, 1, 0.5, 'C2f.4')
        Conv_5 = Conv(self.network, state_dict, C2f_4.get_output(0), width_512,
                      3, 2, 1, 'Conv.5')
        C2f_6 = C2f(self.network, state_dict, Conv_5.get_output(0), width_512,
                    depth_6, True, 1, 0.5, 'C2f.6')
        Conv_7 = Conv(self.network, state_dict, C2f_6.get_output(0),
                      width_1024, 3, 2, 1, 'Conv.7')
        C2f_8 = C2f(self.network, state_dict, Conv_7.get_output(0), width_1024,
                    depth_3, True, 1, 0.5, 'C2f.8')
        SPPF_9 = SPPF(self.network, state_dict, C2f_8.get_output(0),
                      width_1024, width_1024, 5, 'SPPF.9')
        Upsample_10 = self.network.add_resize(SPPF_9.get_output(0))
        assert Upsample_10, 'Add Upsample_10 failed'
        Upsample_10.resize_mode = trt.ResizeMode.NEAREST
        Upsample_10.shape = Upsample_10.get_output(
            0).shape[:2] + C2f_6.get_output(0).shape[2:]
        input_tensors11 = [Upsample_10.get_output(0), C2f_6.get_output(0)]
        Cat_11 = self.network.add_concatenation(input_tensors11)
        C2f_12 = C2f(self.network, state_dict, Cat_11.get_output(0), width_512,
                     depth_3, False, 1, 0.5, 'C2f.12')
        Upsample13 = self.network.add_resize(C2f_12.get_output(0))
        assert Upsample13, 'Add Upsample13 failed'
        Upsample13.resize_mode = trt.ResizeMode.NEAREST
        Upsample13.shape = Upsample13.get_output(
            0).shape[:2] + C2f_4.get_output(0).shape[2:]
        input_tensors14 = [Upsample13.get_output(0), C2f_4.get_output(0)]
        Cat_14 = self.network.add_concatenation(input_tensors14)
        C2f_15 = C2f(self.network, state_dict, Cat_14.get_output(0), width_256,
                     depth_3, False, 1, 0.5, 'C2f.15')
        Conv_16 = Conv(self.network, state_dict, C2f_15.get_output(0),
                       width_256, 3, 2, 1, 'Conv.16')
        input_tensors17 = [Conv_16.get_output(0), C2f_12.get_output(0)]
        Cat_17 = self.network.add_concatenation(input_tensors17)
        C2f_18 = C2f(self.network, state_dict, Cat_17.get_output(0), width_512,
                     depth_3, False, 1, 0.5, 'C2f.18')
        Conv_19 = Conv(self.network, state_dict, C2f_18.get_output(0),
                       width_512, 3, 2, 1, 'Conv.19')
        input_tensors20 = [Conv_19.get_output(0), SPPF_9.get_output(0)]
        Cat_20 = self.network.add_concatenation(input_tensors20)
        C2f_21 = C2f(self.network, state_dict, Cat_20.get_output(0),
                     width_1024, depth_3, False, 1, 0.5, 'C2f.21')
        input_tensors22 = [
            C2f_15.get_output(0),
            C2f_18.get_output(0),
            C2f_21.get_output(0)
        ]
        batched_nms = Detect(self.network, state_dict, input_tensors22,
                             strides, 'Detect.22', reg_max, fp16, iou_thres,
                             conf_thres, topk)
        for o in range(batched_nms.num_outputs):
            self.network.mark_output(batched_nms.get_output(o))


class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()
        self.is_trt10 = not hasattr(model, "num_bindings")
        num_bindings = model.num_io_tensors if self.is_trt10 else model.num_bindings

        names = [model.get_binding_name(i) if not self.is_trt10 else model.get_tensor_name(i)  for i in range(num_bindings)]

        self.bindings: List[int] = [0] * num_bindings
        num_inputs, num_outputs = 0, 0

        if self.is_trt10:
            for i in range(num_bindings):
                if model.get_tensor_mode(model.get_tensor_name(i)) == trt.TensorIOMode.INPUT:
                    num_inputs += 1
                else:
                    num_outputs += 1
        else:
            for i in range(num_bindings):
                if model.binding_is_input(i):
                    num_inputs += 1
                else:
                    num_outputs += 1

        self.num_bindings = num_bindings
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.idx = list(range(self.num_outputs))

    def __init_bindings(self) -> None:
        idynamic = odynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            if not self.is_trt10:
                assert self.model.get_binding_name(i) == name
                dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
                shape = tuple(self.model.get_binding_shape(i))
            else:
                assert self.model.get_tensor_name(i) == name
                dtype = self.dtypeMapping[self.model.get_tensor_dtype(name)]
                self.half = trt.nptype(self.model.get_tensor_dtype(name)) == np.float16
                shape = tuple(self.model.get_tensor_shape(name))
            if -1 in shape:
                idynamic |= True
            inp_info.append(Tensor(name, dtype, shape))
            
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            if not self.is_trt10:
                assert self.model.get_binding_name(i) == name
                dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
                shape = tuple(self.model.get_binding_shape(i))
            else:
                assert self.model.get_tensor_name(i) == name
                dtype = self.dtypeMapping[self.model.get_tensor_dtype(name)]
                shape = tuple(self.model.get_tensor_shape(name))
            if -1 in shape:
                odynamic |= True
            out_info.append(Tensor(name, dtype, shape))

        if not odynamic:
            self.output_tensor = [
                torch.empty(info.shape, dtype=info.dtype, device=self.device)
                for info in out_info
            ]
        self.idynamic = idynamic
        self.odynamic = odynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def set_desired(self, desired: Optional[Union[List, Tuple]]):
        if isinstance(desired,
                      (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> Union[Tuple, torch.Tensor]:
        with torch.cuda.stream(self.stream):
            # Ensure inputs are contiguous and in the right format
            assert len(inputs) == self.num_inputs
            contiguous_inputs = [
                i.contiguous().half() if i.dtype != torch.float16 and self.half else i.contiguous() 
                for i in inputs
            ]

            # Set input tensors
            for i in range(self.num_inputs):
                if self.is_trt10:
                    self.context.set_tensor_address(self.input_names[i], contiguous_inputs[i].data_ptr())
                    if self.idynamic:
                        self.context.set_input_shape(self.input_names[i], tuple(contiguous_inputs[i].shape))
                else:
                    self.bindings[i] = contiguous_inputs[i].data_ptr()
                    if self.idynamic:
                        self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))

            # Prepare output tensors
            outputs = []
            for i in range(self.num_outputs):
                if self.odynamic:
                    shape = (tuple(self.context.get_tensor_shape(self.output_names[i]))
                            if self.is_trt10 else
                            tuple(self.context.get_binding_shape(i + self.num_inputs)))
                    output = torch.empty(size=shape, dtype=self.out_info[i].dtype, device=self.device)
                else:
                    output = self.output_tensor[i]

                if self.is_trt10:
                    self.context.set_tensor_address(self.output_names[i], output.data_ptr())
                else:
                    self.bindings[i + self.num_inputs] = output.data_ptr()
                outputs.append(output)

            # Execute inference
            if self.is_trt10:
                self.context.execute_async_v3(self.stream.cuda_stream)
            else:
                self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)

            # Return outputs
            return tuple(outputs[i] for i in self.idx) if len(outputs) > 1 else outputs[0]


class TRTProfilerV1(trt.IProfiler):

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.total_runtime = 0.0
        self.recorder = defaultdict(float)

    def report_layer_time(self, layer_name: str, ms: float):
        self.total_runtime += ms * 1000
        self.recorder[layer_name] += ms * 1000

    def report(self):
        f = '\t%40s\t\t\t\t%10.4f'
        print('\t%40s\t\t\t\t%10s' % ('layername', 'cost(us)'))
        for name, cost in sorted(self.recorder.items(), key=lambda x: -x[1]):
            print(
                f %
                (name if len(name) < 40 else name[:35] + ' ' + '*' * 4, cost))
        print(f'\nTotal Inference Time: {self.total_runtime:.4f}(us)')


class TRTProfilerV0(trt.IProfiler):

    def __init__(self):
        trt.IProfiler.__init__(self)

    def report_layer_time(self, layer_name: str, ms: float):
        f = '\t%40s\t\t\t\t%10.4fms'
        print(f % (layer_name if len(layer_name) < 40 else layer_name[:35] +
                   ' ' + '*' * 4, ms))
