import builtins
from typing import Any, TypeAlias

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


class OpenCLExecutionContext(builtins.object):
    ...


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

DEVICE_EXEC_KERNEL: int
DEVICE_EXEC_NATIVE_KERNEL: int
DEVICE_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: int
DEVICE_FP_DENORM: int
DEVICE_FP_FMA: int
DEVICE_FP_INF_NAN: int
DEVICE_FP_ROUND_TO_INF: int
DEVICE_FP_ROUND_TO_NEAREST: int
DEVICE_FP_ROUND_TO_ZERO: int
DEVICE_FP_SOFT_FLOAT: int
DEVICE_LOCAL_IS_GLOBAL: int
DEVICE_LOCAL_IS_LOCAL: int
DEVICE_NO_CACHE: int
DEVICE_NO_LOCAL_MEM: int
DEVICE_READ_ONLY_CACHE: int
DEVICE_READ_WRITE_CACHE: int
DEVICE_TYPE_ACCELERATOR: int
DEVICE_TYPE_ALL: int
DEVICE_TYPE_CPU: int
DEVICE_TYPE_DEFAULT: int
DEVICE_TYPE_DGPU: int
DEVICE_TYPE_GPU: int
DEVICE_TYPE_IGPU: int
DEVICE_UNKNOWN_VENDOR: int
DEVICE_VENDOR_AMD: int
DEVICE_VENDOR_INTEL: int
DEVICE_VENDOR_NVIDIA: int
Device_EXEC_KERNEL: int
Device_EXEC_NATIVE_KERNEL: int
Device_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: int
Device_FP_DENORM: int
Device_FP_FMA: int
Device_FP_INF_NAN: int
Device_FP_ROUND_TO_INF: int
Device_FP_ROUND_TO_NEAREST: int
Device_FP_ROUND_TO_ZERO: int
Device_FP_SOFT_FLOAT: int
Device_LOCAL_IS_GLOBAL: int
Device_LOCAL_IS_LOCAL: int
Device_NO_CACHE: int
Device_NO_LOCAL_MEM: int
Device_READ_ONLY_CACHE: int
Device_READ_WRITE_CACHE: int
Device_TYPE_ACCELERATOR: int
Device_TYPE_ALL: int
Device_TYPE_CPU: int
Device_TYPE_DEFAULT: int
Device_TYPE_DGPU: int
Device_TYPE_GPU: int
Device_TYPE_IGPU: int
Device_UNKNOWN_VENDOR: int
Device_VENDOR_AMD: int
Device_VENDOR_INTEL: int
Device_VENDOR_NVIDIA: int
KERNEL_ARG_CONSTANT: int
KERNEL_ARG_LOCAL: int
KERNEL_ARG_NO_SIZE: int
KERNEL_ARG_PTR_ONLY: int
KERNEL_ARG_READ_ONLY: int
KERNEL_ARG_READ_WRITE: int
KERNEL_ARG_WRITE_ONLY: int
KernelArg_CONSTANT: int
KernelArg_LOCAL: int
KernelArg_NO_SIZE: int
KernelArg_PTR_ONLY: int
KernelArg_READ_ONLY: int
KernelArg_READ_WRITE: int
KernelArg_WRITE_ONLY: int
OCL_VECTOR_DEFAULT: int
OCL_VECTOR_MAX: int
OCL_VECTOR_OWN: int
