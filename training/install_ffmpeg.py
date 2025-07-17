import subprocess
import sys


def install_ffmpeg():
    print("Installing ffmpeg installation...")

    subprocess.check_call([sys.executable, "-m", "pip",
         "install","--upgrade", "pip"])
    
    subprocess.check_call([sys.executable, "-m", "pip",
         "install","--upgrade", "setuptools"])

    try:
        subprocess.check_call([sys.executable, "-m", "pip",
             "install","--upgrade", "ffmpeg-python"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing ffmpeg-python: {e}")
    
    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O",
            "/tmp/ffmpeg.tar.xz"
        ])
        
        subprocess.check_call([
            "tar",
            "-xvf",
            "/tmp/ffmpeg.tar.xz",
            "-C",
            "/tmp"
        ])

        result = subprocess.run(
            ["find","/tmp","-name","ffmpeg","-type","f"],
            capture_output=True,
            text=True
        )

        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp",ffmpeg_path,"/usr/local/bin/ffmpeg"])
        subprocess.check_call(["chmod","+x","/usr/local/bin/ffmpeg"])

        print("FFmpeg installed successfully.")
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")


if __name__ == "__main__":
    install_ffmpeg()
