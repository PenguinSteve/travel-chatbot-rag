# RAG Tourism Chatbot API

## 🚀 Chạy Demo

### 1. Cài đặt dependencies

```powershell
pip install -r requirements.txt
```

### 2. Cấu hình file .env

Tạo file `.env` với nội dung:

```env
GROQ_API_KEY=your-groq-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=rag-tourism
RAG_TOP_K=5
LLM_MODEL=llama-3.3-70b-versatile
PORT=8080
```

### 3. Import dữ liệu (chỉ chạy lần đầu)

```powershell
python store_data.py
```

### 4. Chạy server

```powershell
uvicorn main:app --reload --port 8080
```

### 5. Test API

```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:8080/health

# Ask question
$body = @{ query = "Du lịch Hà Nội có gì hay?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8080/ask -ContentType 'application/json' -Body $body
```

**Swagger UI:** http://localhost:8080/docs

### Định dạng câu trả lời tại đây

Chào bạn! Mặc dù hôm nay xe bị hỏng nên bạn không thể ra ngoài, nhưng vẫn có rất nhiều thông tin thú vị về du lịch Việt Nam mà bạn có thể khám phá ngay tại nhà. Dưới đây là một số gợi ý dựa trên những lưu ý và kinh nghiệm trong tài liệu mà tôi đã nhận được:\n\n### 1. Những điểm du lịch nổi bật ở các miền\n| Miền | Địa điểm tiêu biểu | Điểm đặc sắc |\n|------|-------------------|--------------|\n| **Miền Bắc** | Hà Nội (Khu phố cổ, Hồ Hoàn Kiếm, Lăng Bác) | Văn hoá lịch sử, ẩm thực phong phú |\n| | Sa Pa (đỉnh Fansipan, bản làng dân tộc) | Cảnh núi non hùng vĩ, khí hậu mát mẻ |\n| | Hạ Long (vịnh Hạ Long, đảo Đảo Cát Bà) | Vịnh biển đá vôi kỳ quan thế giới |\n| **Miền Trung** | Đà Nẵng (cầu Rồng, Bán đảo Sơn Trà) | Bãi biển đẹp, kiến trúc hiện đại |\n| | Hội An (phố cổ, đèn lồng) | Di sản UNESCO, không gian lãng mạn |\n| | Huế (Kinh thành, lăng tẩm) | Di sản hoàng gia, kiến trúc cổ kính |\n| **Miền Nam** | TP.HCM (Bảo tàng Chứng tích Chiến tranh, Dinh Độc Lập, phố Bùi Viện) | Sự năng động, ẩm thực đường phố |\n| | Cần Thơ (chợ Ninh Kiều, sông Hậu) | Văn hoá miền sông nước, du lịch nông thôn |\n| | Phú Quốc (bãi biển, rừng ngập mặn) | Khu nghỉ dưỡng sang trọng, sinh thái đa dạng |\n\n### 2. Một số lưu ý an toàn khi du lịch\n- **Thời tiết:** Tránh đi vào những ngày thời tiết xấu (mưa gió, biển động) để giảm rủi ro, đặc biệt ở các khu vực ven biển hoặc núi cao.\n- **Phương tiện di chuyển:** Nếu bạn đi xe máy, hãy chắc chắn rằng chỉ sử dụng xe số ở những địa điểm có địa hình dốc và luôn giữ tay lái vững. Tránh đi vào ban đêm ở những nơi ít người vì có thể gặp nguy hiểm.\n- **Dịch vụ ăn uống:** Khi thuê chòi ăn uống, hãy hỏi giá kỹ lưỡng trước khi sử dụng để tránh bất ngờ.\n- **Trang bị cá nhân:** Mang theo áo khoác và kem chống nắng, nhất là khi di chuyển vào buổi trưa hoặc ở những nơi có nắng gắt.\n\n### 3. Kinh nghiệm lên lịch trình tự túc (đặc biệt là Hà Nội)\n- **Chọn thời điểm:** Nếu muốn tránh nhiệt độ cao và cảm giác khó chịu, hãy cân nhắc đi vào các tháng mát hơn (tháng 9‑11 hoặc tháng 12‑2). \n- **Lên kế hoạch chi tiết:** Xác định trước các địa điểm muốn tham quan, sắp xếp thời gian di chuyển hợp lý và chuẩn bị đầy đủ hành trang (đồ dùng cá nhân, thuốc men, áo mưa nếu cần).\n- **Tham khảo nguồn thông tin:** Các sách hướng dẫn như Lonely Planet, blog du lịch, hoặc các diễn đàn du lịch sẽ giúp bạn có cái nhìn tổng quan và những gợi ý thực tế.\n\n### 4. Đọc thêm và khám phá\n- **Blog du lịch cá nhân:** Nhiều du khách chia sẻ trải nghiệm thực tế, ví dụ như cảm nhận về Đà Nẵng và cách người dân địa phương “đưa bạn đi” để tự trải nghiệm.\n- **Video và ảnh:** Xem các video du lịch trên YouTube hoặc Instagram để có hình ảnh sinh động về các địa danh.\n- **Sách hướng dẫn:** “Lonely Planet Vietnam” là một nguồn tham khảo tốt cho những ai muốn lên kế hoạch chi tiết.\n\n### 5. Gợi ý cho buổi học tại nhà\n- **Tìm hiểu văn hoá ẩm thực:** Bạn có thể thử nấu một vài món ăn đặc trưng của từng miền (phở Hà Nội, bánh mì Sài Gòn, mì Quảng Đà Nẵng) và tìm hiểu nguồn gốc, cách chế biến.\n- **Tham gia lớp học trực tuyến:** Nhiều nền tảng giáo dục cung cấp khóa học về lịch sử, văn hoá và du lịch Việt Nam.\n- **Xem phim tài liệu:** Các bộ phim như “Vietnam: A Journey Through Time” hoặc “The Last Days of Saigon” sẽ giúp bạn hiểu sâu hơn về lịch sử và con người Việt Nam.\n\nHy vọng những thông tin trên sẽ giúp bạn có một cái nhìn tổng quan về du lịch Việt Nam và chuẩn bị tốt hơn cho những chuyến đi trong tương lai. Chúc bạn học tập hiệu quả và luôn giữ tinh thần khám phá! 🚀🌏
