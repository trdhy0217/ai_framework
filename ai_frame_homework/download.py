import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/move_boundary_hdnn/"
SAVE_ROOT = os.path.expanduser("./forced_move")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_dir(url, local_path):
    """
    递归遍历 Apache 目录，把每个子目录和文件都下载下来，
    并跳过那些带有 '?' 的排序链接。
    """
    print(f"正在访问目录：{url}")
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for link in soup.find_all("a"):
        name = link.get("href")
        # 1. 跳过上级目录链接 "../"
        if not name or name == "../":
            continue
        # 2. 跳过排序、过滤之类的链接（它们的 href 中会包含 '?C=' 或者其他 '?'）
        if "?" in name:
            continue

        full_url = urljoin(url, name)
        local_target = os.path.join(local_path, name)

        # 如果 href 以 '/' 结尾，就当作子目录
        if name.endswith("/"):
            ensure_dir(local_target)
            download_dir(full_url, local_target)
        else:
            # 普通文件，下载到本地
            ensure_dir(local_path)
            if os.path.exists(local_target):
                print(f"  已跳过（已存在）：{local_target}")
                continue

            print(f"  开始下载文件：{name}")
            with requests.get(full_url, stream=True) as r2:
                r2.raise_for_status()
                with open(local_target, "wb") as f:
                    for chunk in r2.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

if __name__ == "__main__":
    ensure_dir(SAVE_ROOT)
    download_dir(BASE_URL, SAVE_ROOT)
    print("全部下载完成！")
