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
**Xin chào bạn!**  
Dưới đây là lịch trình 5 ngày tại Thành phố Hồ Chí Minh (31/10 – 04/11/2025) dựa trên những thông tin bạn cung cấp. Mỗi ngày được bố trí hợp lý để bạn có thể thưởng thức ẩm thực đường phố, tham quan các điểm nổi bật và tránh xa những thời tiết mưa nặng nhất.

---

## **Ngày 1 – 31/10/2025**  
**Thời tiết:** 23‑28 °C, mưa vừa, khả năng mưa 86 %  
**Sự kiện:** Đến sân bay, check‑in, khám phá Phạm Ngũ Lão & Bến Thành

| Giờ | Hoạt động | Ghi chú |
|-----|-----------|---------|
| 08:00 | Đến sân bay Tan Son Nhat | |
| 09:30 | Check‑in tại **The Reverie Saigon** (sang trọng 5 sao, gần Quận 1) | Đặt phòng trước để tránh giá tăng |
| 10:30 | Đi bộ quanh **Phạm Ngũ Lão** | Tham quan phố cổ, chụp ảnh tại các quán cà phê cổ |
| 12:00 | Ăn trưa tại **Bánh mì Huỳnh Hoa** | Thưởng thức bánh mì truyền thống, ăn nhanh khi di chuyển |
| 13:30 | Tham quan **Bến Thành Market** | Mua sắm quà lưu niệm, trải nghiệm nhịp sống địa phương |
| 15:30 | Thư giãn tại phòng khách sạn, thưởng thức cà phê | |
| 18:00 | Dùng bữa tối tại **Quán Ăn Ngon** | Thử các món ăn đường phố như gỏi cuốn, bún thịt nướng |
| 20:30 | Dạo quanh Quận 1, tham quan **Nhà thờ Đức Bà** | Đèn đường lung linh, không gian yên bình |

---

## **Ngày 2 – 01/11/2025**  
**Thời tiết:** 24‑31 °C, mưa rơi nặng, khả năng mưa 86 %  
**Sự kiện:** Khám phá Quận 1, thưởng thức cơm tấm

| Giờ | Hoạt động | Ghi chú |
|-----|-----------|---------|
| 08:00 | Khởi hành nội bộ (đi bộ hoặc taxi) | Tránh giờ cao điểm |
| 09:00 | Tham quan **Bưu điện Sài Gòn** | Điểm ảnh đẹp, lịch sử |
| 10:30 | Tham quan **Công viên 23/9** (nếu thời tiết cho phép) | Nếu mưa, chuyển sang quán cà phê trong nhà |
| 12:00 | Ăn trưa tại **Cơm Tấm Ba Ghiền** | Thưởng thức cơm tấm đặc trưng, món ăn nhanh |
| 14:00 | Tham quan **Chợ Bến Thành** (tiếp tục) | Mua sắm, thử các món ăn nhẹ |
| 16:00 | Thư giãn tại phòng khách sạn | |
| 18:30 | Dùng bữa tối tại **Bún Chả 145** | Thử bún chả, món ăn đặc trưng của Sài Gòn |
| 20:30 | Dạo quanh Quận 1, thưởng thức cà phê | |

---

## **Ngày 3 – 02/11/2025**  
**Thời tiết:** 23‑28 °C, mưa rơi nặng, khả năng mưa 85 %  
**Sự kiện:** Tham quan các địa điểm trong Quận 1, thưởng thức chè

| Giờ | Hoạt động | Ghi chú |
|-----|-----------|---------|
| 08:30 | Khởi hành nội bộ | |
| 09:00 | Tham quan **Nhà thờ Đức Bà** (nếu chưa đi) | |
| 10:30 | Tham quan **Bưu điện Sài Gòn** (nếu chưa đi) | |
| 12:00 | Ăn trưa tại **Quán Ăn Ngon** (lặp lại) | Thử món mới |
| 14:00 | Tham quan **Chợ Bến Thành** | |
| 16:00 | Thư giãn tại phòng khách sạn | |
| 18:30 | Dùng bữa tối tại **Chè Thái Nguyên** | Thưởng thức chè, tráng miệng |
| 20:30 | Dạo quanh Quận 1, thưởng thức cà phê | |

---

## **Ngày 4 – 03/11/2025**  
**Thời tiết:** Dự kiến tương tự ngày 01/11/02/11 (mưa nặng, 23‑31 °C)  
**Sự kiện:** Khám phá thêm các quán ăn đường phố, thư giãn

| Giờ | Hoạt động | Ghi chú |
|-----|-----------|---------|
| 09:00 | Khởi hành nội bộ | |
| 10:00 | Tham quan **Bến Thành Market** (lặp lại) | |
| 12:00 | Ăn trưa tại **Bánh mì Huỳnh Hoa** (lặp lại) | |
| 14:00 | Tham quan **Quận 1** (đi bộ quanh phố cổ) | |
| 16:00 | Thư giãn tại phòng khách sạn | |
| 18:30 | Dùng bữa tối tại **Cơm Tấm Ba Ghiền** (lặp lại) | |
| 20:30 | Dạo quanh Quận 1, thưởng thức cà phê | |

---

## **Ngày 5 – 04/11/2025**  
**Thời tiết:** Dự kiến tương tự ngày 01/11/02/11 (mưa nặng, 23‑31 °C)  
**Sự kiện:** Chuẩn bị khởi hành, mua sắm cuối cùng

| Giờ | Hoạt động | Ghi chú |
|-----|-----------|---------|
| 08:00 | Check‑out tại khách sạn | |
| 09:00 | Tham quan **Bến Thành Market** (lần cuối) | Mua quà lưu niệm |
| 11:00 | Ăn trưa tại **Bún Chả 145** (lần cuối) | |
| 13:00 | Di chuyển tới sân bay | |
| 15:00 | Rời khỏi Thành phố Hồ Chí Minh | |

---

### **Lưu ý chung**

- **Thời tiết mưa nặng**: Hãy chuẩn bị áo mưa, dù, và ưu tiên các hoạt động trong nhà khi cần.  
- **Di chuyển**: Sử dụng taxi hoặc Grab để tránh tắc đường, đặc biệt vào giờ cao điểm.  
- **Ẩm thực**: Bạn đã có danh sách các quán ăn nổi tiếng; hãy thử các món khác nhau mỗi ngày để trải nghiệm đa dạng hương vị.  
- **Chỗ ở**: The Reverie Saigon cung cấp dịch vụ sang trọng và tiện nghi, nằm ngay trung tâm Quận 1, thuận tiện cho việc di chuyển tới các điểm tham quan. Nếu muốn tiết kiệm, Khách sạn Hương Sen hoặc Windsor Plaza cũng rất hợp lý và gần các địa điểm quan trọng.

---

**Chúc bạn có một chuyến đi tuyệt vời tại Thành phố Hồ Chí Minh!** Nếu cần điều chỉnh lịch trình hoặc thêm thông tin, cứ thoải mái liên hệ nhé. 🌞