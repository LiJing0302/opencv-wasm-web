# 检查 mkcert 是否已安装
if ! command -v mkcert &> /dev/null; then
    echo "mkcert 未安装，正在安装..."
    brew install mkcert  # macOS
    # window 执行以下命令
    # choco install mkcert # Windows 
else
    echo "mkcert 已安装"
fi

# 安装根证书
mkcert -install

# 获取本机局域网 IP
LOCAL_IP=$(ipconfig getifaddr en0)
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(ipconfig getifaddr en1)
fi

echo "本机局域网 IP: $LOCAL_IP"

# 生成证书和私钥文件
mkcert localhost 127.0.0.1 ::1 "$LOCAL_IP"  # 生成证书和私钥文件
