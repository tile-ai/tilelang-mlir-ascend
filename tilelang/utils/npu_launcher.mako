#include "npu_launcher.h"
#define PY_SSIZE_T_CLEAN
% if need_debug:
#define __CCE_ENABLE_PRINT__
% endif
% if debug_print_code:

${debug_print_code}
% endif
% if enable_grid_warn_print:
#define ENABLE_GRID_WARN_PRINT
% endif
#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

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
    aclrtPtrAttributes attributes;
    aclError status = aclrtPointerGetAttributes(ptr_info.dev_ptr, &attributes);

    if (status == ACL_SUCCESS) {
      if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE && attributes.location.type != 4) {
        Py_DECREF(ret);
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
        return ptr_info;
      }
    } else {
      Py_DECREF(ret);
      PyErr_Format(PyExc_RuntimeError,
                   "Failed to query pointer attributes at argument %d. "
                   "Error code: %d. This may indicate invalid memory address "
                   "or NPU device error.",
                   idx, status);
      ptr_info.valid = false;
      return ptr_info;
    }
    Py_DECREF(ret);
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}

static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds, ${arg_decls}) {
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattened 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  void *workspace_addr_ptr = NULL;
  uint32_t blockNum4Workspace = gridX * gridY * gridZ;
  % if workspace_size > 0:
  uint64_t totalWorkSpaceSize = ${workspace_size} * blockNum4Workspace;
  workspace_addr_ptr = const_cast<void *>(at::empty(totalWorkSpaceSize, at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte)).storage().data());
  % endif
  % if enable_taskqueue:
  auto launch_call = [&]() -> rtError_t {
  % endif
    uint32_t blockNum = gridX * gridY * gridZ;

  % if enable_grid_warn_print:
    #ifdef ENABLE_GRID_WARN_PRINT
      static bool warned = false;
      if (!warned && blockNum > (uint32_t)${num_physical_blocks}) {
        printf("WARNING: Grid %u > physical limit ${num_physical_blocks}, performance maybe reduced.\\n",blockNum);
        warned = true;
    }
    #endif
  % endif
  % if enable_auto_map_parallel_blocks:
    blockNum = std::min(blockNum, (uint32_t)${num_physical_blocks});
  % endif
    // set mixBlockNumRation for nodeBasicBlockDim for msprof report
    uint32_t mixBlockNumRation = ${mix_block_dim_ratio};
    uint32_t nodeBasicBlockDim = (mixBlockNumRation << 16) + blockNum;

  % if need_debug:
    cce::internal::DebugTunnelData *DTData = cce::internal::DebugTunnel::Open(blockNum);
  % endif
    rtError_t ret = RT_ERROR_NONE;
  % if target_support_ffts:
    void *ffts_addr = NULL;
    uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) {
      return% if enable_taskqueue: ret% endif;
    }
  % endif
    // stub argument for workspace
    void *syncBlockLock_ptr = NULL;
    uint16_t ModuleId = 0;
  % if lock_num > 0:
    uint64_t syncBlockLockSize = ${lock_num} * sizeof(int64_t);
    syncBlockLock_ptr = const_cast<void *>(at_npu::native::allocate_workspace(syncBlockLockSize, stream).storage().data());
    if (!syncBlockLock_ptr) {
      % if enable_taskqueue:
      return ret;
      % else:
      fprintf(stderr, "Error: syncBlockLock allocation failed\\n"); return;
      % endif
    }
    std::vector<int64_t> lockInitData(${lock_num}, ${lock_ini_val});
    ret = rtMemcpy(
        syncBlockLock_ptr, syncBlockLockSize,
        reinterpret_cast<void *>(lockInitData.data()), syncBlockLockSize,
        RT_MEMCPY_HOST_TO_DEVICE
    );
    if (ret != RT_ERROR_NONE) {
      return% if enable_taskqueue: ret% endif;
    }
  % endif
  % if workspace_size > 0:
    if (ret != RT_ERROR_NONE) {
      return% if enable_taskqueue: ret% endif;
    }
  % endif
    struct __attribute__((packed)) {
  % if target_support_ffts:
      void* ffts_addr __attribute__((aligned(8)));
  % endif
  % if not force_simt_only:
      void* syncBlockLock __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
  % endif
      % for i, ty in signature.items():
      % if i not in constants:
      ${ty_to_cpp[ty]} arg${i} __attribute__((aligned(${4 if ty[0] != '*' and ty[-2:] != '64' else 8})));
      % endif
      % endfor
      % for mark, ty in grid_info.items():
      ${ty_to_cpp[ty]} grid${mark} __attribute__((aligned(4)));
      % endfor
      % if need_debug:
      void* DTData __attribute__((aligned(8)));
      % endif
    } args = {
  % if target_support_ffts:
      static_cast<void*>(ffts_addr),
  % endif
  % if not force_simt_only:
      % if lock_num > 0:
      static_cast<void*>(syncBlockLock_ptr),
      % else:
      nullptr,
      % endif
      % if workspace_size > 0:
      static_cast<void*>(workspace_addr_ptr),
      % else:
      nullptr,
      % endif
  % endif
      % for i, ty in signature.items():
      % if i not in constants:
      static_cast<${ty_to_cpp[ty]}>(arg${i}),
      % endif
      % endfor
      % for mark, ty in grid_info.items():
      static_cast<${ty_to_cpp[ty]}>(grid${mark}),
      % endfor
      % if need_debug:
      , static_cast<void*>(DTData)
      % endif
    };
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
  % if compile_on_910_95 and enable_simt:
    rtArgsEx_t argsInfo = {};
    argsInfo.args = static_cast<void*>(&args);
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo = {};
    cfgInfo.localMemorySize = ${shared_mem_dynamic_size};
    ret = rtKernelLaunchWithFlagV2(func, blockNum, &argsInfo, NULL, stream, 0, &cfgInfo);
  % else:
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);
  % endif
  % if need_debug:
    void *&stream_ref = const_cast<void*&>(stream);
    cce::internal::DebugTunnel::Close(DTData, stream_ref);
  % endif
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
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
    }
    if (__MsprofFlagL1)
    {
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = ${task_type};
      nodeBasicInfo.data.nodeBasicInfo.blockDim = nodeBasicBlockDim;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

  % if is_mix_task_type:
      // 'mix' kernel need to report the ctxID
      MsprofAdditionalInfo info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
      info.threadId = threadId;
      info.timeStamp = endTime;
      MsprofContextIdInfo ctxId;
      ctxId.opName = opNameHashID;
      ctxId.ctxIdNum = 1;
      for (uint32_t i = 0; i < ctxId.ctxIdNum; i++) {
        ctxId.ctxIds[i] = i;
      }
      size_t copyLen = sizeof(MsprofContextIdInfo);
      if (copyLen > MSPROF_ADDTIONAL_INFO_DATA_LENGTH) {
        copyLen = MSPROF_ADDTIONAL_INFO_DATA_LENGTH;
      }
      memcpy(info.data, &ctxId, copyLen);
      MsprofReportAdditionalInfo(false, static_cast<void *>(&info), sizeof(MsprofAdditionalInfo));
  % endif

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
      if (tensorShapes.size() > 0) {
        % for i, ty in signature.items():
        % if ty.startswith("*") and i < 5:
        dataTypes[${i}] = ${sigtype_to_int[ty[1:]]};
        % endif
        % endfor
      }
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {
        auto fillTensorData = [&](int index, int tensorType) {
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            profTensorData->tensorData[index].shape[j] = 0;
          }
        };
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }
      }
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }
  % if enable_taskqueue:
    return ret;
  % else:
    ret = rtStreamSynchronize(stream);
  % endif
  };
  % if enable_taskqueue:
  at_npu::native::OpCommand cmd;
  cmd.Name(name.c_str()).SetCustomHandler(launch_call).Run();
  % endif
  return;
}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_shape(PyObject *tensor) {
  std::vector<int64_t> shape;

  // Early return if tensor is None or null
  if (!tensor || tensor == Py_None) {
    return shape;
  }

  // Calling tensor.size()
  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {
    return shape;
  }
  // Using PySequence_Fast to improve access efficiency
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {
        shape.push_back(PyLong_AsLong(dim));
      }
    }
  }
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;

  % for i, ty in signature.items():
  ${extracted_ty[ty]} _arg${i};
  % endfor
  if(!PyArg_ParseTuple(
      args, "${format_str}",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      % if len(signature) > 0:
      % for i, ty in signature.items():
      , &_arg${i}
      % endfor
      % endif
      )
    ) {
    return NULL;
  }
  if (__MsprofFlagL1)
  {
    % for i, ty in signature.items():
    % if ty[0] == "*":
    { auto tmp = _get_tensor_shape(_arg${i}); if (!tmp.empty()) tensorShapes.push_back(tmp); }
    % endif
    % endfor
  }

  if (launch_enter_hook != Py_None){
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }
  }

  // raise exception asap
  % for i, ty in signature.items():
  % if ty[0] == "*":
  DevicePtrInfo ptr_info${i} = getPointer(_arg${i}, ${i}); if (!ptr_info${i}.valid) return NULL;
  % endif
  % endfor
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds, ${launch_args});
  if (PyErr_Occurred()) {
    return NULL;
  }
  if(launch_exit_hook != Py_None){
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }
  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__tilelang_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___tilelang_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h
  return m;
}
