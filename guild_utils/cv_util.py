import re
import subprocess


def convert_cuda_visible_devices(env):
    cuda_visible_devices = env.get("CUDA_VISIBLE_DEVICES", "")
    uuid_cuda_visible_devices = map_cuda_visible_devices(cuda_visible_devices)
    env["CUDA_VISIBLE_DEVICES"] = uuid_cuda_visible_devices
    return env


def parse_gpu_list(output):
    gpu_mapping = {}
    pattern = re.compile(r"GPU (\d+): .* \(UUID: (GPU-[a-f0-9-]+)\)")
    matches = pattern.findall(output)
    for match in matches:
        gpu_index = int(match[0])
        uuid = match[1]
        gpu_mapping[gpu_index] = uuid
    return gpu_mapping


def convert_to_uuid_list(cuda_visible_devices, gpu_mapping):
    cuda_visible_devices = cuda_visible_devices.split(",")
    uuid_list = []

    for item in cuda_visible_devices:
        if item.startswith("GPU-"):
            uuid_list.append(item)
        else:
            gpu_index = int(item)
            if gpu_index in gpu_mapping:
                uuid_list.append(gpu_mapping[gpu_index])
            else:
                print(f"GPU index {gpu_index} not found in mapping.")
    return uuid_list


def map_cuda_visible_devices(cuda_visible_devices):
    nvidia_smi_output = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    # Parse GPU list and create mapping using regex
    gpu_mapping = parse_gpu_list(nvidia_smi_output)
    # Convert to UUID list
    uuid_list = convert_to_uuid_list(cuda_visible_devices, gpu_mapping)
    return ",".join(uuid_list)
