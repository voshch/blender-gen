
import os
import sys
import click
import subprocess
import grequests as requests
import socket

sys.path.append(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "src"))

from util import Target, Mode, Log

log = Log()


def configure(python_cmd, mode):
    log.print(f"RUNNING CONFIGURE STEP:\n")
    proc = subprocess.Popen([python_cmd, "./src/configure.py", "--mode_internal",
                            mode, *sys.argv[1:]], stdout=log.stdout, stderr=log.stderr)
    return proc.wait()


def render():
    if os.path.isfile("/data/intermediate/render/render.lock"):
        os.remove("/data/intermediate/render/render.lock")

    log.print(f"RUNNING RENDER STEP:\n")
    proc = subprocess.Popen(["blender", "--background", "--python",
                            "./src/render.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    proc.wait()

    if os.path.isfile("/data/intermediate/render/render.lock"):
        os.remove("/data/intermediate/render/render.lock")
        return 0

    return 1


def merge(python_cmd, mode):
    log.print(f"RUNNING MERGE STEP:\n")
    proc = subprocess.Popen([python_cmd, "./src/merge.py", "--mode_internal",
                            mode, *sys.argv[1:]], stdout=log.stdout, stderr=log.stderr)
    return proc.wait()


def postfx(python_cmd, mode):
    log.print(f"RUNNING POSTFX STEP:\n")
    proc = subprocess.Popen([python_cmd, "./src/postfx.py", "--mode_internal",
                            mode, *sys.argv[1:]], stdout=log.stdout, stderr=log.stderr)
    return proc.wait()


def run(taskid, target, endpoint, mode):
    python_cmd = os.path.realpath(sys.executable)

    log.print(f"!!RUNNING BLENDER-GEN IN MODE {mode}\n")

    match target:
        case Target.Configure:
            status = configure(python_cmd, mode)
            if status != 0:
                raise RuntimeError(
                    f"configure step failed with {status}")

        case Target.Render:
            status = render()
            if status != 0:
                raise RuntimeError(
                    f"render step failed with {status}")

        case Target.Merge:
            status = merge(python_cmd, mode)
            if status != 0:
                raise RuntimeError(
                    f"merge step failed with {status}")

        case Target.PostFX:
            status = postfx(python_cmd, mode)
            if status != 0:
                raise RuntimeError(
                    f"postfx step failed with {status}")

        case _:
            raise ValueError(f"unknown target {target}")


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--mode", type=click.Choice(["all", Mode.Train, Mode.Val]), default="all", help="all|train|val create training or validation images")
@click.option("--target", default="all")
@click.option("--endpoint", default=None, help="http endpoint for sending current progress")
@click.option("--taskID", default="", help="task ID")
@click.option("--output", type=click.Choice(["shell", "file"]), default="shell", help="output to stdout or to /data/output/log.txt")
def main(mode, target, endpoint, taskid, output):

    os.makedirs("/data/intermediate/config/", exist_ok=True)
    with open("/data/intermediate/config/log.conf", "w") as f:  # for blender python script
        f.write(output)

    if output == "file":
        os.makedirs("/data/log/", exist_ok=True)
        # the easiest way to create or truncate
        open("/data/log/stdout.txt", "w").close()
        log.stdout = open("/data/log/stdout.txt", "a")
        open("/data/log/stderr.txt", "w").close()
        log.stderr = open("/data/log/stderr.txt", "a")

    target = [Target.Configure, Target.Render, Target.Merge,
              Target.PostFX] if target == "all" else target.split(",")
    
    mode = [Mode.Train, Mode.Val] if mode == "all" else [mode]

    try:
        for current_target in target:
            for current_mode in mode:
                run(taskid, current_target, endpoint, current_mode)
                log.print("\n")

    except RuntimeError as e:
        if endpoint != None:
            requests.post(f"{endpoint}/task/stop",
                          json=dict(taskId=taskid)).send()

        # https://stackoverflow.com/a/45532289
        log.err(e["message"] if hasattr(e, "message") else repr(e))
        raise e

    else:
        if endpoint != None:
            requests.post(f"{endpoint}/task/finish",
                          json=dict(taskId=taskid)).send()

        log.print("finished successfully")

    finally:
        if output == "file":
            log.stdout.close()
            log.stderr.close()


if __name__ == "__main__":
    main()
