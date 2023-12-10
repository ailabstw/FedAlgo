import subprocess
import logging
import os
import jax

def jax_cpu_cores():
    return len(jax.devices('cpu'))

def bash(command, *args, **kargs):
    PopenObj = subprocess.Popen(
        command,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        preexec_fn = os.setsid,
        shell = True,
        executable = "/bin/bash",
        *args, **kargs
    )
    out, err = PopenObj.communicate()
    out = out.decode("utf8").rstrip("\r\n").split("\n")
    err = err.decode("utf8").rstrip("\r\n").split("\n")
    if PopenObj.returncode != 0:
        logging.error(f"command failed")
        logging.error(command)
        for i in err:
            logging.error(i)
        raise RuntimeError
    return out, err