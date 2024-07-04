import builtins
from typing import Any, Final, TypeAlias

retval: TypeAlias = Any

class Device(builtins.object):
    def OpenCLVersion(self) -> retval:
        """"""

    def OpenCL_C_Version(self) -> retval:
        """

        See help(type(self)) for accurate signature.

        """

    def addressBits(self) -> retval:
        """"""

    def available(self) -> retval:
        """"""

    def compilerAvailable(self) -> retval:
        """"""

    def deviceVersionMajor(self) -> retval:
        """"""

    def deviceVersionMinor(self) -> retval:
        """"""

    def doubleFPConfig(self) -> retval:
        """"""

    def driverVersion(self) -> retval:
        """"""

    def endianLittle(self) -> retval:
        """"""

    def errorCorrectionSupport(self) -> retval:
        """"""

    def executionCapabilities(self) -> retval:
        """"""

    def extensions(self) -> retval:
        """"""

    def globalMemCacheLineSize(self) -> retval:
        """"""

    def globalMemCacheSize(self) -> retval:
        """"""

    def globalMemCacheType(self) -> retval:
        """"""

    def globalMemSize(self) -> retval:
        """"""

    def halfFPConfig(self) -> retval:
        """"""

    def hostUnifiedMemory(self) -> retval:
        """"""

    def image2DMaxHeight(self) -> retval:
        """"""

    def image2DMaxWidth(self) -> retval:
        """"""

    def image3DMaxDepth(self) -> retval:
        """"""

    def image3DMaxHeight(self) -> retval:
        """"""

    def image3DMaxWidth(self) -> retval:
        """"""

    def imageFromBufferSupport(self) -> retval:
        """"""

    def imageMaxArraySize(self) -> retval:
        """"""

    def imageMaxBufferSize(self) -> retval:
        """"""

    def imageSupport(self) -> retval:
        """"""

    def intelSubgroupsSupport(self) -> retval:
        """"""

    def isAMD(self) -> retval:
        """"""

    def isExtensionSupported(self, extensionName) -> retval:
        """"""

    def isIntel(self) -> retval:
        """"""

    def isNVidia(self) -> retval:
        """"""

    def linkerAvailable(self) -> retval:
        """"""

    def localMemSize(self) -> retval:
        """"""

    def localMemType(self) -> retval:
        """"""

    def maxClockFrequency(self) -> retval:
        """"""

    def maxComputeUnits(self) -> retval:
        """"""

    def maxConstantArgs(self) -> retval:
        """"""

    def maxConstantBufferSize(self) -> retval:
        """"""

    def maxMemAllocSize(self) -> retval:
        """"""

    def maxParameterSize(self) -> retval:
        """"""

    def maxReadImageArgs(self) -> retval:
        """"""

    def maxSamplers(self) -> retval:
        """"""

    def maxWorkGroupSize(self) -> retval:
        """"""

    def maxWorkItemDims(self) -> retval:
        """"""

    def maxWriteImageArgs(self) -> retval:
        """"""

    def memBaseAddrAlign(self) -> retval:
        """"""

    def name(self) -> retval:
        """"""

    def nativeVectorWidthChar(self) -> retval:
        """"""

    def nativeVectorWidthDouble(self) -> retval:
        """"""

    def nativeVectorWidthFloat(self) -> retval:
        """"""

    def nativeVectorWidthHalf(self) -> retval:
        """"""

    def nativeVectorWidthInt(self) -> retval:
        """"""

    def nativeVectorWidthLong(self) -> retval:
        """"""

    def nativeVectorWidthShort(self) -> retval:
        """"""

    def preferredVectorWidthChar(self) -> retval:
        """"""

    def preferredVectorWidthDouble(self) -> retval:
        """"""

    def preferredVectorWidthFloat(self) -> retval:
        """"""

    def preferredVectorWidthHalf(self) -> retval:
        """"""

    def preferredVectorWidthInt(self) -> retval:
        """"""

    def preferredVectorWidthLong(self) -> retval:
        """"""

    def preferredVectorWidthShort(self) -> retval:
        """"""

    def printfBufferSize(self) -> retval:
        """"""

    def profilingTimerResolution(self) -> retval:
        """"""

    def singleFPConfig(self) -> retval:
        """"""

    def type(self) -> retval:
        """"""

    def vendorID(self) -> retval:
        """"""

    def vendorName(self) -> retval:
        """"""

    def version(self) -> retval:
        """"""

    def getDefault(self) -> retval:
        """"""

class OpenCLExecutionContext(builtins.object): ...

def Device_getDefault() -> retval:
    """
    .
    """

def finish() -> None:
    """
    .
    """

def haveAmdBlas() -> retval:
    """
    .
    """

def haveAmdFft() -> retval:
    """
    .
    """

def haveOpenCL() -> retval:
    """
    .
    """

def setUseOpenCL(flag) -> None:
    """
    .
    """

def useOpenCL() -> retval:
    """
    .
    """

DEVICE_EXEC_KERNEL: Final[int]
DEVICE_EXEC_NATIVE_KERNEL: Final[int]
DEVICE_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: Final[int]
DEVICE_FP_DENORM: Final[int]
DEVICE_FP_FMA: Final[int]
DEVICE_FP_INF_NAN: Final[int]
DEVICE_FP_ROUND_TO_INF: Final[int]
DEVICE_FP_ROUND_TO_NEAREST: Final[int]
DEVICE_FP_ROUND_TO_ZERO: Final[int]
DEVICE_FP_SOFT_FLOAT: Final[int]
DEVICE_LOCAL_IS_GLOBAL: Final[int]
DEVICE_LOCAL_IS_LOCAL: Final[int]
DEVICE_NO_CACHE: Final[int]
DEVICE_NO_LOCAL_MEM: Final[int]
DEVICE_READ_ONLY_CACHE: Final[int]
DEVICE_READ_WRITE_CACHE: Final[int]
DEVICE_TYPE_ACCELERATOR: Final[int]
DEVICE_TYPE_ALL: Final[int]
DEVICE_TYPE_CPU: Final[int]
DEVICE_TYPE_DEFAULT: Final[int]
DEVICE_TYPE_DGPU: Final[int]
DEVICE_TYPE_GPU: Final[int]
DEVICE_TYPE_IGPU: Final[int]
DEVICE_UNKNOWN_VENDOR: Final[int]
DEVICE_VENDOR_AMD: Final[int]
DEVICE_VENDOR_INTEL: Final[int]
DEVICE_VENDOR_NVIDIA: Final[int]
Device_EXEC_KERNEL: Final[int]
Device_EXEC_NATIVE_KERNEL: Final[int]
Device_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: Final[int]
Device_FP_DENORM: Final[int]
Device_FP_FMA: Final[int]
Device_FP_INF_NAN: Final[int]
Device_FP_ROUND_TO_INF: Final[int]
Device_FP_ROUND_TO_NEAREST: Final[int]
Device_FP_ROUND_TO_ZERO: Final[int]
Device_FP_SOFT_FLOAT: Final[int]
Device_LOCAL_IS_GLOBAL: Final[int]
Device_LOCAL_IS_LOCAL: Final[int]
Device_NO_CACHE: Final[int]
Device_NO_LOCAL_MEM: Final[int]
Device_READ_ONLY_CACHE: Final[int]
Device_READ_WRITE_CACHE: Final[int]
Device_TYPE_ACCELERATOR: Final[int]
Device_TYPE_ALL: Final[int]
Device_TYPE_CPU: Final[int]
Device_TYPE_DEFAULT: Final[int]
Device_TYPE_DGPU: Final[int]
Device_TYPE_GPU: Final[int]
Device_TYPE_IGPU: Final[int]
Device_UNKNOWN_VENDOR: Final[int]
Device_VENDOR_AMD: Final[int]
Device_VENDOR_INTEL: Final[int]
Device_VENDOR_NVIDIA: Final[int]
KERNEL_ARG_CONSTANT: Final[int]
KERNEL_ARG_LOCAL: Final[int]
KERNEL_ARG_NO_SIZE: Final[int]
KERNEL_ARG_PTR_ONLY: Final[int]
KERNEL_ARG_READ_ONLY: Final[int]
KERNEL_ARG_READ_WRITE: Final[int]
KERNEL_ARG_WRITE_ONLY: Final[int]
KernelArg_CONSTANT: Final[int]
KernelArg_LOCAL: Final[int]
KernelArg_NO_SIZE: Final[int]
KernelArg_PTR_ONLY: Final[int]
KernelArg_READ_ONLY: Final[int]
KernelArg_READ_WRITE: Final[int]
KernelArg_WRITE_ONLY: Final[int]
OCL_VECTOR_DEFAULT: Final[int]
OCL_VECTOR_MAX: Final[int]
OCL_VECTOR_OWN: Final[int]
