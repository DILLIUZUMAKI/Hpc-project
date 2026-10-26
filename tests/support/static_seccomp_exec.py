"""Delivery-test helper: execute a statically linked test binary under seccomp.

No file-opening or networking syscalls; no fork/vfork; clone only for threads.
Not part of the application's supported Docker execution backend.
"""
import ctypes as C
import ctypes.util
import os
import resource
import sys

binary, args = sys.argv[1], sys.argv[2:]
fd = os.open(binary, os.O_RDONLY | os.O_CLOEXEC)
libc = C.CDLL(None, use_errno=True)
sec = C.CDLL(ctypes.util.find_library('seccomp'), use_errno=True)
sec.seccomp_init.argtypes=[C.c_uint32]; sec.seccomp_init.restype=C.c_void_p
sec.seccomp_syscall_resolve_name.argtypes=[C.c_char_p]; sec.seccomp_syscall_resolve_name.restype=C.c_int
sec.seccomp_rule_add.argtypes=[C.c_void_p,C.c_uint32,C.c_int,C.c_uint]
sec.seccomp_load.argtypes=[C.c_void_p]; sec.seccomp_load.restype=C.c_int

class Arg(C.Structure):
    _fields_=[('arg',C.c_uint),('op',C.c_uint),('a',C.c_uint64),('b',C.c_uint64)]

ALLOW=0x7fff0000
ctx=sec.seccomp_init(0x00050000 | 1) # all unspecified syscalls return EPERM
def allow(name,*filters,action=ALLOW):
    number=sec.seccomp_syscall_resolve_name(name.encode())
    if number>=0:
        result=sec.seccomp_rule_add(ctx,action,number,len(filters),*filters)
        if result: raise RuntimeError((name,result))

for name in ['read','write','close','fstat','lseek','mmap','mprotect','munmap','brk','rt_sigaction','rt_sigprocmask',
             'rt_sigreturn','sigaltstack','arch_prctl','set_tid_address','set_robust_list','rseq','futex','futex_waitv',
             'sched_yield','clock_gettime','clock_getres','nanosleep','clock_nanosleep','getpid','getppid','gettid',
             'getuid','geteuid','getgid','getegid','getrandom','madvise','exit','exit_group','restart_syscall']:
    allow(name)
allow('sched_getaffinity',Arg(0,4,0,0))
allow('sched_setaffinity',Arg(0,4,0,0))
allow('prlimit64',Arg(0,4,0,0),Arg(2,4,0,0))
allow('tgkill',Arg(0,4,os.getpid(),0))
allow('clone',Arg(0,7,0x10900,0x10900)) # CLONE_VM | CLONE_SIGHAND | CLONE_THREAD
allow('clone3',action=0x00050000 | 38) # glibc falls back to clone
allow('execveat',Arg(0,4,fd,0),Arg(4,4,4096,0))
# fstat via newfstatat on stdout/stderr only, not arbitrary path lookups.
for outfd in [1,2]: allow('newfstatat',Arg(0,4,outfd,0),Arg(3,4,4096,0))

argv=(C.c_char_p*(len(args)+2))(*[x.encode() for x in ['/program',*args]],None)
environment={'PATH':'','OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS','2'), 'OMP_DYNAMIC':'FALSE',
             'OMP_THREAD_LIMIT':'4','OMP_PROC_BIND':'FALSE','LC_ALL':'C'}
envp=(C.c_char_p*(len(environment)+1))(*[f'{k}={v}'.encode() for k,v in environment.items()],None)
resource.setrlimit(resource.RLIMIT_CPU,(5,5))
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_NPROC,(32,32))
resource.setrlimit(resource.RLIMIT_NOFILE,(16,16))
if libc.prctl(38,1,0,0,0)!=0: raise OSError('no_new_privs failed')
if sec.seccomp_load(ctx)!=0: raise OSError('seccomp setup failed')
# This is the only exec permitted; the binary FD is close-on-exec. No host files,
# sockets, device opens, mount changes, ptrace, process forks or further opens.
libc.syscall(322,fd,C.c_char_p(b''),argv,envp,4096)
os._exit(127)
