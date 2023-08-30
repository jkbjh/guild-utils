import contextlib
import ctypes
import os
import signal
import subprocess
import time
from tempfile import TemporaryDirectory

libc = ctypes.CDLL("libc.so.6")
PR_SET_PDEATHSIG = 1


def set_pdeathsig(sig=signal.SIGTERM):
    def callable():
        os.setsid()
        return libc.prctl(PR_SET_PDEATHSIG, sig)

    return callable


@contextlib.contextmanager
def make_mps_controller():
    with TemporaryDirectory() as temp_dir:
        mps_controller = MPSController(temp_dir)
        try:
            mps_controller.start()
            yield mps_controller
            mps_controller.command("quit")
        finally:
            mps_controller.kill()


class MPSController:
    ALLOWED_COMMANDS = {"quit", "get_server_list"}

    def __init__(self, directory):
        self.pipe_dir = directory
        self.process = None

    def start(self):
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        # self.process = subprocess.Popen(["nvidia-cuda-mps-control", "-f"], env=env, preexec_fn=os.setsid)
        self.process = subprocess.Popen(["nvidia-cuda-mps-control", "-f"], env=env, preexec_fn=set_pdeathsig)
        # wait for the server to start.
        for i in range(5):
            try:
                subprocess.check_call("echo get_server_list | nvidia-cuda-mps-control", env=env, shell=True)
            except subprocess.SubprocessError:
                time.sleep(0.1)
                continue
            break

    def command(self, cmd):
        assert cmd in self.ALLOWED_COMMANDS
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
        return subprocess.check_output(f"echo {cmd} | nvidia-cuda-mps-control", env=env, shell=True)

    def kill(self):
        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        self.process.wait()

    def get_env_keys(self):
        return {"CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir}
