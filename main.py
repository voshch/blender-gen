import os
import sys
import click
import subprocess
import grequests as requests


def configure(python_cmd):
    proc = subprocess.Popen([python_cmd, "./src/configure.py",
                            *sys.argv[1:]], stdout=sys.stdout, stderr=subprocess.PIPE)
    return proc.wait(), proc.stderr.read()


def render():
    with open(os.devnull, "w") as devnull:
        proc = subprocess.Popen(["blender", "--background", "--python",
                                "./src/render.py"], stdout=devnull, stderr=subprocess.PIPE)
        return proc.wait(), proc.stderr.read()


def merge(python_cmd):
    proc = subprocess.Popen([python_cmd, "./src/merge.py", *
                            sys.argv[1:]], stdout=sys.stdout, stderr=subprocess.PIPE)
    return proc.wait(), proc.stderr.read()


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--target", type=click.Choice(["all", "configure", "render", "merge"]), default="all")
@click.option("--endpoint", default=None, help="http endpoint for sending current progress")
@click.option("--taskID", default="", help="task ID")
def main(target, endpoint, taskid):

    python_cmd = os.path.realpath(sys.executable)

    try:
        if target in ["all", "configure"]:
            status, stderr = configure(python_cmd)
            if status != 0:
                with open("/data/output/error.log", "ab") as f:
                    f.write("\n[STEP CONFIGURE]:\n".encode("ascii"))
                    f.write(stderr)
                raise RuntimeError(
                    f"configure step failed with {status}. Log has been written to /data/output/error.log")

        if target in ["all", "render"]:
            status, stderr = render()
            if status != 0:
                with open("/data/output/error.log", "ab") as f:
                    f.write("\n[STEP RENDER]:\n".encode("ascii"))
                    f.write(stderr)
                raise RuntimeError(
                    f"render step failed with {status}. Log has been written to /data/output/error.log")

        if target in ["all", "merge"]:
            status, stderr = merge(python_cmd)
            if status != 0:
                with open("/data/output/error.log", "ab") as f:
                    f.write("\n[STEP MERGE]:\n".encode("ascii"))
                    f.write(stderr)
                raise RuntimeError(
                    f"merge step failed with {status}. Log has been written to /data/output/error.log")

    except RuntimeError as e:
        if endpoint != None:
            requests.post(f"{endpoint}/error", data=dict(id=taskid)).send()

        raise e

    else:
        if endpoint != None:
            requests.post(f"{endpoint}/done", data=dict(id=taskid)).send()

        print("finished successfully")


if __name__ == "__main__":
    main()
