# Thử thách nâng cao: index toàn bộ Economics 2e

## Nguồn
- OpenStax, *Principles of Economics 2e* (bản web col12122/1.4), 974 trang, giấy phép CC BY 4.0 (được chia sẻ lại, cần ghi nguồn).
- Tải bản gốc tại https://openstax.org/details/books/principles-economics-2e (bản 2e, số trang theo PDF 974 trang). Không nộp PDF kèm bài; đường dẫn PDF đặt qua biến môi trường `ADV_PDF` (mặc định: `Economics2e.pdf` trong thư mục Day 6 `starter/data`).
- `source_id` = `openstax-econ2e-full`; số trang = chỉ số trang PDF, bắt đầu từ 1 (giống benchmark).

## Dựng lại index
Chạy từ thư mục gốc bài lab, cần OpenAI key (đọc từ `.env` Day 6 qua `src/day7/config.py`):

```
python -m advanced.index_text                    # text từng trang -> data/chunks_text.json
python -m advanced.index_visual                  # vision model -> data/chunks_visual.json (resume được, cache ở data/cache/vision)
python -m advanced.index_visual --pages 517,736  # chỉ chạy thử vài trang
python -m advanced.build_index                   # gộp -> data/chunks.json, embed vào advanced/chroma_db, ghi data/index_stats.json
```

## Nội dung được index
- **Text:** mọi trang, bỏ dòng chân trang và đầu trang lặp lại, cắt ~1200 ký tự (chồng ~150) trong phạm vi một trang. Metadata: `page`, `chapter` (từ mục lục PDF, 0 = phần đầu/cuối sách), `chapter_title`, `kind="text"`.
- **Nội dung trực quan:** trang có ảnh raster, từ 15 đường vector trở lên, hoặc có dòng bắt đầu bằng "Table N.N"/"Figure N.N". Mỗi trang được render 150 dpi và gửi cho `gpt-4.1-mini` cùng text của trang; kết quả là bảng (markdown), biểu đồ (loại, trục, đơn vị, giá trị đọc ra), sơ đồ (nhãn, mũi tên), ảnh (một dòng). Mỗi mục là một chunk có `kind`, `region` (ví dụ "Figure 30.8") và `caption_bbox` (vị trí chú thích trên trang, để mở đúng vùng ở PDF gốc khi kiểm chứng).
- Thống kê (số trang, số mục theo loại, lỗi, chi phí, thời gian): `data/index_stats.json`; lỗi từng trang: `data/visual_failures.json`.

## Giới hạn đã biết
- Nội dung trực quan do model tạo ra, không tự bảo đảm đúng: giá trị biểu đồ chỉ là ước lượng ("about"), bảng lớn có thể thiếu hoặc sai ô. Cần đối chiếu bản gốc khi chấm.
- `caption_bbox` chỉ là vị trí dòng chú thích (lần khớp đầu tiên của nhãn), không phải ảnh crop; mục "unlabelled image" hoặc nhãn có hậu tố như "(a)" có thể không có bbox.
- Trang chỉ có hình trang trí bị bỏ nếu mô tả quá ngắn (< 40 ký tự).
- Tách trang theo câu nên bảng trong text bị nối thành một dòng; bản transcript bảng nằm ở chunk trực quan.
- Chunk không đi qua ranh giới trang, nên đoạn/bảng kéo dài nhiều trang bị chia.
