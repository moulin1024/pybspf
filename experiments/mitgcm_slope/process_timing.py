"""Time one child process and sample CPU/thread use without changing affinity."""
import resource
import subprocess
import time
import psutil


def run_logged(command, *, cwd, env, log):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    counts, thread_cpu, errors = [], {}, set()
    start = time.perf_counter()
    with open(log, 'w') as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                threads = process.threads()
                counts.append(len(threads))
                for thread in threads:
                    thread_cpu[thread.id] = max(thread_cpu.get(thread.id, 0.), thread.user_time+thread.system_time)
            except psutil.NoSuchProcess:
                break
            except psutil.Error as exc:
                errors.add(type(exc).__name__)
            try:
                child.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                pass
        returncode = child.wait()
    wall = time.perf_counter()-start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    return dict(process_wall_seconds=wall, returncode=returncode,
        process_cpu_seconds=cpu, mean_cpu_cores_used=cpu/wall,
        max_sampled_os_threads=max(counts, default=None),
        sampled_threads_with_over_0p05_cpu_seconds=sum(v>.05 for v in thread_cpu.values()),
        thread_cpu_seconds={str(k):v for k,v in thread_cpu.items()},
        monitoring_errors=sorted(errors), sampling_interval_seconds=.1)
