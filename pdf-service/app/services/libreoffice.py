import subprocess
import os
import platform

def libreoffice_convert(file):
    output_dir = "data/pdf"
    
    # 根据操作系统选择 LibreOffice 路径
    if platform.system() == "Windows":
        # 默认的 LibreOffice 安装路径（可根据实际情况调整）
        libreoffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"
    else:
        # Linux 系统使用 soffice 命令
        libreoffice_path = "soffice"
    
    cmd = [
        libreoffice_path,
        "--headless",
        "--convert-to",
        "pdf",
        file,
        "--outdir",
        output_dir
    ]
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    subprocess.run(cmd, check=True)
    
    pdf = os.path.join(
        output_dir,
        os.path.basename(file).split(".")[0] + ".pdf"
    )
    
    return pdf