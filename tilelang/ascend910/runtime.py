"""Launch/load source generation for the Ascend 910 (A2) runtime.

The generator is byte-compatible with the legacy ``jit_npu`` implementation:
the emitted C++ is part of the golden corpus and is compared directly.
"""

from __future__ import annotations

import os
import re


def get_bisheng_path() -> str:
    """Return the CANN bisheng compiler used by the debug tunnel headers."""
    for key in ("TILELANG_BISHENG_PATH", "BISHENG_PATH"):
        value = os.environ.get(key)
        if value and os.path.isfile(value):
            return value
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        root = os.environ.get(key)
        if not root:
            continue
        for rel in ("tools/ccec_compiler/bin/bisheng", "bin/bisheng"):
            candidate = os.path.join(root, rel)
            if os.path.isfile(candidate):
                return candidate
    return "bisheng"


class NPUUtils:
    """Small capability provider; overridable by the JIT for non-24-core parts."""

    _instance = None
    aicore_num = 24
    aivector_num = 48

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_aicore_num(self):
        return self.aicore_num

    def get_aivector_core_num(self):
        return self.aivector_num


def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        # Boolean
        "i1": 12,  # BOOL
        # Integer types
        "i8": 2,  # INT8
        "i16": 6,  # INT16
        "i32": 3,  # INT32
        "i64": 9,  # INT64
        # Unsigned integer types
        "u32": 8,  # UINT32
        "u64": 10,  # UINT64
        # Floating point types
        "fp16": 1,  # FLOAT16
        "bf16": 27,  # DT_BF16
        "fp32": 0,  # FLOAT
        "fp64": 11,  # DOUBLE
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")

    return MAP_SIGTYPE_TO_INT[sigty]


def extract_device_print_code_from_cann():
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(
        ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib"
    )

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), "r") as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = " ".join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = "\n".join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer(r"\[aicore\]", code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos - 1, -1, -1):
                if (
                    src[i] == "}"
                ):  # this relies on that all [aicore] functions come after normal functions
                    left = i + 1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == "{":
                    brace_nest += 1
                elif src[j] == "}":
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ""
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += code[segment_start:]

        # remove __gm__ and rename macros
        new_code = new_code.replace("__gm__", " ")
        new_code = new_code.replace("__CCELIB_RT_ERROR_NONE", "RT_ERROR_NONE")
        new_code = new_code.replace("__CCELIB_RT_MEMORY_HBM", "RT_MEMORY_HBM")
        new_code = new_code.replace(
            "__CCELIB_RT_MEMCPY_HOST_TO_DEVICE", "RT_MEMCPY_HOST_TO_DEVICE"
        )
        new_code = new_code.replace(
            "__CCELIB_RT_MEMCPY_DEVICE_TO_HOST", "RT_MEMCPY_DEVICE_TO_HOST"
        )
        return new_code

    # the following headers should be included in this order
    return "\n".join(
        [
            read_header("common/common_impl.h"),
            read_header("internal/debug_tunnel/payload.h"),
            read_header("internal/debug_tunnel/payload_impl.h"),
            read_header("internal/debug_tunnel/tunnel.h"),
            read_header("internal/debug_tunnel/tunnel_impl.h"),
        ]
    )


def generate_npu_wrapper_src(
    constants,
    signature,
    workspace_size,
    mix_mode,
    lock_num,
    lock_ini_val,
    need_debug,
    force_simt_only=False,
    shared_mem_dynamic_size=0,
    compile_on_910_95=False,
    target_support_ffts=True,
):
    def _ty_to_cpp(ty):
        if ty[0] == "*":
            return "void*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _extracted_ty(ty):
        if ty[0] == "*":
            return "PyObject*"
        return {
            "i1": "int32_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    arg_decls = ", ".join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    """
    args:
        int gridX, gridY, gridZ;
        rtStream_t stream;
        const void *function;
        PyObject* packed_metadata, *launch_metadata;
        PyObject* launch_enter_hook, *launch_exit_hook;
        *args_expand
    """
    format = "iiiKKOOOO" + "".join(
        [_format_of(_extracted_ty(ty)) for ty in signature.values()]
    )

    grid_info = {"X": "i32", "Y": "i32", "Z": "i32"}

    enable_taskqueue = os.getenv("TILELANG_ENABLE_TASKQUEUE", "true").lower() in (
        "true",
        "1",
    )
    enable_auto_map_parallel_blocks = False
    npu_utils = NPUUtils.get()
    num_physical_blocks = (
        npu_utils.get_aivector_core_num()
        if mix_mode == "aiv"
        else npu_utils.get_aicore_num()
    )
    task_type = (
        "MSPROF_GE_TASK_TYPE_AIV"
        if mix_mode == "aiv"
        else "MSPROF_GE_TASK_TYPE_AI_CORE"
    )
    LINE_CHANGE_CHAR = chr(10)  # it is \n

    cpp_device_pointer = """
typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}
"""

    cpp_msprof_extern = """
extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}
"""

    cpp_msprof_callback = """
  MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h
"""

    cpp_msprof_call_before_launch = """
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }
"""

    cpp_msprof_call_after_launch = f"""
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {{
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }}
    if (__MsprofFlagL1)
    {{
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = {task_type};
      nodeBasicInfo.data.nodeBasicInfo.blockDim = blockNum;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {{
        {
        LINE_CHANGE_CHAR.join(
            f"dataTypes[{i}] = {convert_sigtype_to_int(ty[1:])};"
            for i, ty in signature.items()
            if ty.startswith("*") and i < 5
        )
    }
      }}
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {{
        auto fillTensorData = [&](int index, int tensorType) {{
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {{
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }}
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {{
            profTensorData->tensorData[index].shape[j] = 0;
          }}
        }};
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }}
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){{
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }}
      }}
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }}
"""

    # Capture by value: the handler is enqueued into torch_npu's task queue and
    # may run on another thread after _launch() has returned.
    cpp_launch_lambda_decl = (
        "auto launch_call = [=]() mutable -> rtError_t" if enable_taskqueue else ""
    )

    # Dispatch.  The downstream (non-910_95) path keeps RunOpApi(..., sync=true)
    # on purpose: that trailing synchronisation drains the task queue and syncs
    # the stream before the buffers prepared above are released at the end of
    # _launch.  With the queue enabled the handler runs asynchronously on the
    # consumer thread, so without it a queued launch could still read buffers
    # that the calling thread has already handed back to the allocator.
    cpp_launch_dispatch = (
        "at_npu::native::OpCommand cmd; cmd.Name(name.c_str())"
        ".SetCustomHandler(launch_call).Run();"
        if (enable_taskqueue and compile_on_910_95)
        else (
            "at_npu::native::OpCommand::RunOpApi(name.c_str(), launch_call, true);"
            if enable_taskqueue
            else ""
        )
    )

    return f"""
#include "npu_launcher.h"
#define PY_SSIZE_T_CLEAN
{"#define __CCE_ENABLE_PRINT__" if need_debug else ""}
{extract_device_print_code_from_cann() if need_debug else ""}

#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

{cpp_msprof_extern}

{cpp_device_pointer}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds, {
        arg_decls
    }) {{
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattened 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  // Buffers are prepared here, on the *calling* thread, before the handler is
  // built.  The handler may run on torch_npu's task-queue consumer thread, and
  // touching the caching / workspace allocators from there re-enters the same
  // bounded task queue and deadlocks.  The tensor handles live in this scope, so
  // they are created and released on the calling thread as well.
  rtError_t ret = RT_ERROR_NONE;
  rtError_t prep_ret = RT_ERROR_NONE;
  uint32_t blockNum = gridX * gridY * gridZ;
  {
        "blockNum = std::min(blockNum, (uint32_t)" + str(num_physical_blocks) + ");"
        if enable_auto_map_parallel_blocks
        else ""
    }
  void *syncBlockLock = NULL;
  void *workspace_addr = NULL;
  at::Tensor syncBlockLockTensor;
  at::Tensor workspaceTensor;
  {
        '''
  void *ffts_addr = NULL;
  uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
  if (ret != RT_ERROR_NONE) {
    prep_ret = ret;
  }
  '''
        if target_support_ffts
        else ""
    }
  {
        f'''
  if (prep_ret == RT_ERROR_NONE) {{
    uint64_t syncBlockLockSize = {lock_num} * sizeof(int64_t);
    syncBlockLockTensor = at_npu::native::allocate_workspace(syncBlockLockSize, stream);
    syncBlockLock = const_cast<void *>(syncBlockLockTensor.storage().data());
    if (syncBlockLock == nullptr) {{
      fprintf(stderr, "Error: syncBlockLock allocation failed");
      prep_ret = 1;
    }} else {{
      std::vector<int64_t> lockInitData({lock_num}, {lock_ini_val});
      ret = rtMemcpy(syncBlockLock, syncBlockLockSize, reinterpret_cast<void *>(lockInitData.data()),
                     syncBlockLockSize, RT_MEMCPY_HOST_TO_DEVICE);
      if (ret != RT_ERROR_NONE) {{
        prep_ret = ret;
      }}
    }}
  }}
  '''
        if lock_num > 0
        else ""
    }
  {
        f'''
  if (prep_ret == RT_ERROR_NONE) {{
    uint64_t totalWorkSpaceSize = {workspace_size} * blockNum;
    workspaceTensor = at::empty({{static_cast<int64_t>(totalWorkSpaceSize)}},
                                at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte));
    workspace_addr = const_cast<void *>(workspaceTensor.storage().data());
    if (workspace_addr == nullptr) {{
      fprintf(stderr, "Error: workspace allocation failed");
      prep_ret = 1;
    }}
  }}
  '''
        if workspace_size > 0
        else ""
    }
    {cpp_launch_lambda_decl} {{
    if (prep_ret != RT_ERROR_NONE) {{
      return {"prep_ret" if enable_taskqueue else ""};
    }}
    {
        "cce::internal::DebugTunnelData *DTData = cce::internal::DebugTunnel::Open(blockNum);"
        if need_debug
        else ""
    }
    struct __attribute__((packed)) {{
      {"void* ffts_addr __attribute__((aligned(8)));" if target_support_ffts else ""}
      {
        "void* syncBlockLock __attribute__((aligned(8)));"
        if not force_simt_only
        else ""
    }
      {
        "void* workspace_addr __attribute__((aligned(8)));"
        if not force_simt_only
        else ""
    }
      {
        " ".join(
            f"{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != '*' and ty[-2:] != '64' else 8})));"
            for i, ty in signature.items()
            if i not in constants
        )
    }
      {
        " ".join(
            f"{_ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));"
            for mark, ty in grid_info.items()
        )
    }
      {"void* DTData __attribute__((aligned(8)));" if need_debug else ""}
    }} args = {{
      {"static_cast<void*>(ffts_addr)," if target_support_ffts else ""}
      {
        ("static_cast<void*>(syncBlockLock)," if lock_num > 0 else "nullptr,")
        if not force_simt_only
        else ""
    }
      {
        ("static_cast<void*>(workspace_addr)," if workspace_size > 0 else "nullptr,")
        if not force_simt_only
        else ""
    }
      {
        ", ".join(
            f"static_cast<{_ty_to_cpp(ty)}>(arg{i})"
            for i, ty in signature.items()
            if i not in constants
        )
    },
      {
        ", ".join(
            f"static_cast<{_ty_to_cpp(ty)}>(grid{mark})"
            for mark, ty in grid_info.items()
        )
    }
      {", static_cast<void*>(DTData)" if need_debug else ""}
    }};
    {cpp_msprof_call_before_launch}
    {
        f'''
    rtArgsEx_t argsInfo = {{}};
    argsInfo.args = static_cast<void*>(&args);
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo = {{}};
    cfgInfo.localMemorySize = {shared_mem_dynamic_size};
    ret = rtKernelLaunchWithFlagV2(func, blockNum, &argsInfo, NULL, stream, 0, &cfgInfo);
    '''
        if compile_on_910_95
        else "ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);"
    }
    {"void *&stream_ref = const_cast<void*&>(stream);" if need_debug else ""}
    {"cce::internal::DebugTunnel::Close(DTData, stream_ref);" if need_debug else ""}
    {cpp_msprof_call_after_launch}
    {
        "return ret;"
        if enable_taskqueue
        else ("ret = rtStreamSynchronize(stream);" if compile_on_910_95 else "")
    }
   }};
   {cpp_launch_dispatch}
  return;
}}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_shape(PyObject *tensor) {{
  std::vector<int64_t> shape;

  // Early return if tensor is None or null
  if (!tensor || tensor == Py_None) {{
    return shape;
  }}

  // Calling tensor.size()
  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {{
    return shape;
  }}
  // Using PySequence_Fast to improve access efficiency
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {{
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {{
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {{
        shape.push_back(PyLong_AsLong(dim));
      }}
    }}
  }}
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;
  {" ".join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{format}\",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      {
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    }
      )
    ) {{
    return NULL;
  }}
  if (__MsprofFlagL1)
  {{
    {
        LINE_CHANGE_CHAR.join(
            f"{{ auto tmp = _get_tensor_shape(_arg{i}); if (!tmp.empty()) tensorShapes.push_back(tmp); }}"
            for i, ty in signature.items()
            if ty[0] == "*"
        )
    }
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {{
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {{
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }}
  }}

  // raise exception asap
  {
        "; ".join(
            [
                f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
                if ty[0] == "*"
                else ""
                for i, ty in signature.items()
            ]
        )
    };
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds, {
        ", ".join(
            f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"_arg{i}"
            for i, ty in signature.items()
        )
    });
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}
  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__tilelang_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___tilelang_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  {cpp_msprof_callback}
  return m;
}}
"""




def get_npu_launcher_header() -> str:
    """Return the packaged pybind launcher header."""
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1] / "utils" / "npu_launcher.h")
