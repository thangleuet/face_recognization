import os
import re
import sys
import subprocess
from enum import IntEnum, unique
from time import sleep

#
# Copyright 2022 by Vmio System JSC
# All rights reserved.
# Utility functions cpu and process managerment
#

def get_available_cpu():
    """
    Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program

    Returns:
        int: Return number of available cpus. Return -1 if can not determine number of CPUs on this system
    """    
    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return int(res)
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    return -1


@unique
class platform(IntEnum):
    UNKNOW = 0
    LINUX = 1
    WINDOWS = 2
    WINDOWS_CYGWIN = 3
    DARWIN = 4

def check_platform() -> platform:
    """
    Return platform of this system

    Returns:
        Platform: Platform type
    """    
    plf = sys.platform
    if plf.startswith('linux'):
        return platform.LINUX
    if plf.startswith('win32'):
        return platform.WINDOWS
    if plf.startswith('cygwin'):
        return platform.WINDOWS_CYGWIN
    if plf.startswith('darwin'):
        return platform.DARWIN
    return platform.UNKNOW

def assign_process_to_cores(pID: int, cores: int or list[int]) -> bool:
    """
    CPU pinning, allows the user to assign a process to use only a few cores.
    Technically you can bind and unbind a process or thread to CPU or CPUs which here can be termed as CPU cores. 

    Args:
        pID (int): Process ID of the process you want to assign
        cores (int or list[int]): core index or an array of indexes

    Returns:
        bool: True if sucess
    """ 
    handle = None

    try:
        plf = check_platform()
        mask = None
        if isinstance(cores, int):
            mask = 1 << cores
        else:
            if isinstance(cores, list):
                mask = 0x00
                for core_idx in cores:
                    mask = mask | 1 << core_idx # Eg. Cores 0, 1, 5 --> mask = 100011
        
        if mask is None:
            return False
                  
        # For Linux
        if plf == platform.LINUX: #https://man7.org/linux/man-pages/man1/taskset.1.html
            os.system(f"taskset -p {mask} {pID}")
            return True
        
        # For Windows
        # Require PyWin32 package
        if plf == platform.WINDOWS: #https://linustechtips.com/topic/591933-guide-hyper-threading-and-windows-explained-for-real
            import win32api, win32con, win32process
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pID)
            win32process.SetProcessAffinityMask(handle, mask)
            return True
    except Exception as e:
        print(e)
    
    finally:
        if handle is not None:
            win32api.CloseHandle(handle)
        
    return False

if __name__ == "__main__":
    cpus = get_available_cpu()
    if cpus > 0:
        print(f'CPUs: {cpus}')
    else:
        print("Can not determine number of CPUs on this system")
        
    print(f"Patform = {check_platform().name}")
    
    assign_process_to_cores(os.getpid(), 0)
    
    sleep(60)
    