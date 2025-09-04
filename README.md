# 🌟 Nông Trí AI - Chatbot Nông Nghiệp Thông Minh 🌱

## 📌 Giới thiệu

**Nông Trí AI** là một hệ thống hỗ trợ kiến thức nông nghiệp về cây cà phê, sầu riêng và hồ tiêu. Hệ thống sử dụng công nghệ **RAG (Retrieval-Augmented Generation)** kết hợp với các mô hình ngôn ngữ lớn để trích xuất và tổng hợp thông tin chính xác từ kho dữ liệu.

## Dự án bao gồm

### Chatbot Nông Trí AI
Chatbot được xây dựng bằng mô hình RAG giúp người dùng tra cứu thông tin nông nghiệp chính xác và nhanh chóng.
- **Github**: [DieppDiepp/NongTriAI](https://github.com/DieppDiepp/NongTriAI)

### Ứng dụng Nông Trí AI (APK)
Phiên bản di động giúp người dùng truy cập thông tin dễ dàng hơn.
- **Github**: [Zile228/NongTri_App](https://github.com/Zile228/NongTri_App)

## Giao diện ứng dụng
![Nong Tri Web3 & AI Ideathon](https://github.com/DieppDiepp/NongTriAI/raw/main/image/Nông%20Trí%20–%20Chatbot%20trợ%20lý%20nông%20nghiệp%20AI%20dành%20riêng%20cho%20nông%20dân%20Việt%20Nam.png)

----------------------
## Dự án đạt giải Á Quân của cuộc thi 𝐖𝐞𝐛𝟑 & 𝐀𝐈 𝐈𝐝𝐞𝐚𝐭𝐡𝐨𝐧
𝐖𝐞𝐛𝟑 & 𝐀𝐈 𝐈𝐝𝐞𝐚𝐭𝐡𝐨𝐧 là cuộc thi toàn quốc về ý tưởng công nghệ thu hút 1012 đơn đăng ký và 452 ý tưởng được gửi về

![Nong Tri Web3 & AI Ideathon](https://github.com/DieppDiepp/NongTriAI/raw/main/image/top-2.jpg)

-------------
## 🛠️ Kiến trúc hệ thống

1. 🚀 **FastAPI**: Framework backend chính dùng để xây dựng API giao tiếp với chatbot.
2. 📊 **ChromaDB**: Vector database lưu trữ embedding các tài liệu nông nghiệp.
3. 🤖 **Gemini 2.5 Flash**: Mô hình AI từ Google dùng để sinh câu trả lời.
4. 📝 **Embedding Tiếng Việt**: Sử dụng mô hình [dangvantuan/vietnamese-embedding](https://huggingface.co/dangvantuan/vietnamese-embedding) tối ưu cho dữ liệu tiếng Việt.
5. 🦜️🔗 **LangChain + Langsmith**: Sử dụng framework LangChain, Langsmith để tương tác và theo dõi LLMs, áp dụng các kỹ thuật tối ưu như multi-branch, parallel-branch để nâng cao hiệu suất.

![Nong Tri Web3 & AI Ideathon](https://github.com/DieppDiepp/NongTriAI/raw/main/image/langsmith_1.png)

![Nong Tri Web3 & AI Ideathon](https://github.com/DieppDiepp/NongTriAI/raw/main/image/langsmith_2.png)

## Các điểm tối ưu 

1. 📌 **Tích hợp Embedding Tiếng Việt**:
   - Dùng mô hình embedding tiêu chuẩn cho tiếng Việt (được triển khai trong file `VietnameseEmbedding.py`).
   - Tối ưu chiếu dài embedding bằng mean-pooling với mask tín hiệu.

2. 🔎 **Phân nhánh quy trình (Route Prompt)**:
   - Nhận diện câu hỏi thông qua prompt phân loại (định nghĩa trong `NongtriPrompt.py`).
   - Tự động quyết định truy vấn CSDL phù hợp.

3. 🔗 **RAG Chain Linh Hoạt**:
   - Chuỗi pipeline linh hoạt từ truy vấn độc lập đến tổng hợp nội dung (định nghĩa trong `test_gemini.py` và `NongTriConservation.py`).

4. 🎯 **Truy vấn chính xác cao**:
   - Sử dụng k=5 và áp dụng kỹ thuật multi queries để sinh ra 3 phiên bản câu hỏi của người dùng, trích xuất 15 kết quả gần nhất, .
   - Quy định nghiêm ngặt về nguồn trích dẫn và ngôn ngữ trả lời.

## Cách chạy dự án

### 1. 📋 Yêu cầu môi trường

- Python >= 3.10.11
- Cài đặt các package yêu cầu:

```bash
# Window
git clone https://github.com/DieppDiepp/NongTriAI
py -3.10 -m venv venv310
venv310\Scripts\activate
python --version   # Python 3.10.11
pip install -r requirements.txt

# Linux - Thông thường có sẵn python, nhưng phiên bản có thể không phù hợp
# Cài phiên bản python cụ thể
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y
python3.10 --version # Python 3.10.xx thì đã cài thành công 

# Kích hoạt môi trường ảo
git clone https://github.com/DieppDiepp/NongTriAI
python3.10 -m venv venv310
source venv310/bin/activate
pip install -r requirements.txt
```

- Chạy file `showlib.py` để kiểm tra các package/ lib cài thành công 
### 2. 🔐 Cài đặt biến môi trường

Tạo file `.env` với nội dung (Lưu ý file .env nằm cùng cấp với các Scripts python):

```
GOOGLE_API_KEY=your-google-api-key # Lấy trên Google AI Studio 
LANGSMITH_TRACING=true 
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your-langsmith-api-key" # Lấy trên Langsmith
LANGSMITH_PROJECT="your-langsmith-name" # Lấy trên Langsmith
```

```bash
# Window tạo file .env đơn giản
# Linux
touch .env
nano .env # Sau đó dán API KEY vào
```
### 3. 🧑‍💻 Chạy API
Chọn 1 trong 2 cách sau (ưu tiên cách 2)
```bash
tmux new -s fastapi # Tạo tạo session ảo bên trong VPS, tách biệt với terminal, khi tắt terminal session SSH mất, nhưng cái tmux session vẫn chạy trên VPS.

python FastApiDev.py
```

```bash
tmux new -s fastapi # Tạo tạo session ảo bên trong VPS, tách biệt với terminal, khi tắt terminal session SSH mất, nhưng cái tmux session vẫn chạy trên VPS.

uvicorn FastApiDev:app --host 127.0.0.1 --port 8000 --reload # Mỗi khi có thay đổi trong code thì tự động reboost lại hệ thống với thay đổi mới.
```

#### WINDOW
Mở trình duyệt tại: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000docs)

#### LINUX - Test API bằng curl (kết quả trả về chưa được format)
```bash
curl -X POST "http://127.0.0.1:8000/chat" \
-H "Content-Type: application/json" \
-d '{"question":"Bón phân cho cây cà phê vào giai đoạn nào?"}'
```

### 5. Dùng tunel để tạo public address (Ngrok/ Cloudflared tunnel)

#### WINDOW
Tải về và cài đặt Cloudflared tunnel tại: https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.msi

```bash
tmux new -s tunnel # Tạo tạo session ảo bên trong VPS, tách biệt với terminal, khi tắt terminal session SSH mất, nhưng cái tmux session vẫn chạy trên VPS.
cloudflared tunnel --url http://127.0.0.1:8000
```
#### LINUX
```bash
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
cloudflared --version # cloudflared version xxxx -> thành công

cloudflared tunnel --url http://127.0.0.1:8000
```

Sau đó ta sẽ nhận được một thông báo tương tự:

Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  https://comp-entire-risks-alto.trycloudflare.com

Ta thêm hậu tố /docs vào: https://comp-entire-risks-alto.trycloudflare.com/docs -> Đây chính là public IP có thể truy cập được từ tất cả các thiết bị.

Để cố định liên kết truy cập trên của Cloudflared, tìm hiểu thêm tại https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/

### 7.  Kiểm tra các tiến trình đang chạy
```bash
tmux ls
# fastapi: 1 windows (created Thu Sep  4 08:56:45 2025)
# tunnel: 1 windows (created Thu Sep  4 08:57:17 2025)

# Có thể vào lại bằng tmux attach -t fastapi/tunnel
```

## 🌐 Website Demo

1️⃣ Website giới thiệu sản phẩm: [https://nongtri.netlify.app/](https://nongtri.netlify.app/)

![Demo Web](https://github.com/DieppDiepp/NongTriAI/raw/main/image/demoweb.jpg)

2️⃣ Website demo chatbot RAG: [https://nongtrichat.netlify.app/](https://nongtrichat.netlify.app/)
![Demo Web chatbot](https://github.com/DieppDiepp/NongTriAI/blob/main/image/nongtrichat_full.png)

3️⃣ Ứng dụng APK được Lê Vy thành viên trong team dự thi Web3 AI Hackfest phát triển - https://github.com/Zile228/NongTri_App 

-------------
## Hiện tại phiên bản cung cấp thông tin 3 loại cây trồng: cà phê, hồ tiêu, sầu riêng đi kèm với các kỹ thuật tối ưu như multi queries, routing database, structured LLMs response,... đảm bảo tốc độ truy vấn cao và chính xác
**Vì lý do bảo mật, phân đoạn tạo vector store tự động được bảo mật**, nếu có nhu cầu sử dụng vui lòng liên hệ!

## 📂 Cấu trúc thư mục

```
.
├── VietnameseEmbedding.py    # Xử lý embedding tiếng Việt
├── FastApiDev.py             # Khởi chạy FastAPI
├── main_processing.py        # Tích hợp Gemini- flash 2.5
├── formatdocs.py             # Định dạng dữ liệu trả về
├── NongTriConservation.py    # Triển khai chuỗi RAG
├── NongtriPrompt.py          # Định nghĩa prompt
├── structured_response.py    # Ép LLm trả về kết quả cho trước
├── mapping.py                # Chuyển plan_type về lại tiếng Việt
├── Requirements.txt          # Các thư viện yêu cầu 
├── .env                      # API Key cần thiết
└── DB/                       # ChromaDB lưu trữ dữ liệu
```

## 🤝 Góp ý

Hãy tạo pull request hoặc issue nếu bạn muốn đóng góp hoặc báo lỗi!

