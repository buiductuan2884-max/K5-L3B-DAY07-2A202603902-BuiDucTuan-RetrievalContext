# Báo cáo Day 7

Bùi Đức Tuấn · 2A202603902 · L3B

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Phần cơ bản: OpenStax *Principles of Economics 2e*, chương 3–5 (PDF trang 55–136), giấy phép CC BY 4.0. Dùng lại đúng 260 chunk của chiến lược B ở Day 6, cắt theo đoạn và cho phép nối qua trang.
  - Phần nâng cao: toàn bộ cuốn sách, 974 trang, 3.841 chunk. Chi tiết ở mục nâng cao.
- **Vector store, embedding model, phiên bản và số chiều:**
  - Chroma 1.5.9 `PersistentClient`, đặt tại `../K5-L3-Day06-DataIngestionEngineering/starter/chroma_db`, collection `strategy_B`, metric cosine.
  - Embedding là OpenAI `text-embedding-3-small`, 1536 chiều, cùng một model cho chunk và câu hỏi.
  - Day 7 đọc thẳng từ Chroma, không ingest lại. File `data/day7/chunks.json` là bản export của collection này, dùng cho BM25 và để chấm điểm.
- **Agent model, cách chạy và phân chia dev/test:**
  - Phần cơ bản dùng `gpt-4o-mini` với tool calling, `temperature 0`.
  - Dev gồm 43 câu: toàn bộ 40 câu Day 6 (đều có đáp án) và 3 câu mới thiếu evidence.
  - Test gồm 11 câu mới: 8 câu có đáp án và 3 câu thiếu evidence, viết và khóa bằng sha256 trước khi chạy bất cứ thứ gì.
  - Test Day 6 không dùng lại làm test, vì ở Day 6 tôi đã nhìn kết quả test khi chọn chiến lược. 11 câu test Day 6 được chuyển vào dev.
  - Lệnh chạy ở mục Agent.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án: kiểm tra bằng script rằng `evidence` xuất hiện nguyên văn trên trang PDF và nằm trọn trong một chunk. Kết quả 48/48 câu.
  - Câu thiếu evidence: tìm toàn văn trang 55–136 theo từ khóa chính và đọc từng chỗ xuất hiện. Ví dụ "2022" chỉ xuất hiện trong dự báo nhu cầu y tá ở p.93. Lý do ghi trong trường `absence_reason`.
  - Chấm câu trả lời theo thang 0/1/2 của BENCHMARK.md. Chấm citation bằng cách đọc chunk được trích.

## Retrieval và thử nghiệm

Cùng corpus, index và nhãn. Hit@1 và Hit@3 đo trên câu hỏi gốc, không viết lại, và chỉ tính câu có đáp án. "Hit" nghĩa là đúng nguồn, đúng trang, và evidence nằm trọn trong chunk. Context Evidence Rate (CER) là tỷ lệ câu có evidence trong context cuối gửi cho model. Context budget đo bằng ký tự, tính cả dòng nguồn.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `agent_dense_baseline` | dense Chroma, top-3, không filter, 1 lượt search, 2.400 ký tự, giữ chunk theo thứ tự đến | Mốc so sánh | dev | 22/40 (55%) | 29/40 (72,5%) | 26/40 (65%) | 3,15 s/câu; $0,012 cho 43 câu |
| Phương pháp 2: BM25 | BM25 tự viết (k1 1,5, b 0,75), top-10 | Từ khóa chính xác ("consumer surplus") có thể khớp tốt hơn dense | dev | 19/40 (47,5%) | 28/40 (70%) | (chỉ đo retrieval) | 0,03 s cho 40 câu; $0 |
| Phương pháp 3: hybrid `agent_hybrid` | RRF (k=60) của dense top-20 và BM25 top-20, top-3, tối đa 3 search, 2.400 ký tự | Hai cách tìm bù lỗi cho nhau; cho agent search lại | dev | 24/40 (60%) | 30/40 (75%) | 27/40 (67,5%) | 3,34 s/câu; $0,014 |
| Cải thiện 1 `agent_hybrid_score` | hybrid, top-5, 3 search, 3.600 ký tự, chọn context theo score trên mọi lần search | Policy cũ giữ chunk đến trước nên chunk tốt ở lần search sau bị loại vì hết budget | dev | 24/40 | 30/40 | 31/40 (77,5%) | 3,18 s/câu; $0,015 |
| Cải thiện 2 `agent_hybrid_score_filter` | như trên + filter `chapter` do người dùng cho (7/40 câu dev có) | Lọc đúng chương giúp câu có khái niệm trùng ở nhiều chương | dev | 24/40 | 32/40 (80%) | 32/40 (80%) | 3,44 s/câu; $0,015 |
| (Thử, không chọn) `agent_hybrid_score_verify` | như Cải thiện 1 + bước kiểm tra grounding bằng một lần gọi model riêng | Chặn câu trả lời không được evidence hỗ trợ | dev | 24/40 | 30/40 | 31/40 | 5,16 s/câu; $0,020 |
| Cấu hình đã chọn | `agent_hybrid_score_filter` | | test | 6/8 | 8/8 | 8/8 (100%) | 3,58 s/câu; $0,0042 cho 11 câu |
| Baseline | `agent_dense_baseline` | | test | 7/8 | 7/8 | 6/8 (75%) | 2,99 s/câu; $0,0025 cho 11 câu |

Hit@1/Hit@3 của dòng agent là retrieval một lượt bằng câu hỏi gốc với cùng phương pháp. Hybrid có filter ở dev là 24/40 và 32/40, MRR@10 0,725. Dense có filter là 22/40 và 30/40. Trên test, BM25 được 6/8 và 7/8, hybrid 6/8 và 8/8.

**Tiêu chí chọn, ghi trước khi chạy test:**
1. Mọi câu thiếu evidence ở dev phải được từ chối.
2. Nhiều câu có đáp án nhất mà agent trả lời và context cuối chứa evidence gold.
3. Nếu hòa thì xét CER.
4. Nếu vẫn hòa thì chọn cấu hình rẻ và nhanh hơn.

Kết quả: cả 5 cấu hình đều qua điều kiện 1. `agent_hybrid_score_filter` được 32/40, hybrid_score và verify 31, hybrid 27, baseline 26. Tôi chọn `agent_hybrid_score_filter`.

**Hai lỗi dev:**
- **Lỗi 1: citation sai định dạng.** Ở lần chạy dev đầu, mỗi cấu hình có 7–12 câu có đáp án bị chuyển thành từ chối với sự kiện `invalid_citations`, dù chunk được trích có trong context và đúng nội dung (q01, q04, q22).
  - Nguyên nhân: model điền danh sách `citations` nhưng không viết `[chunk-id]` ngay trong câu trả lời, trong khi bộ kiểm tra bắt buộc cả hai.
  - Cách sửa: prompt có ví dụ citation inline, và agent được thêm một lượt sửa với thông báo lỗi cụ thể. Nếu lần hai vẫn sai thì từ chối như cũ.
  - Trước và sau: q01, q04, q09, q22, t20 từ 0/5 lên 5/5 được trả lời. Số câu có đáp án được trả lời ở baseline tăng từ 21/40 lên 36/40.
- **Lỗi 2: câu tiếng Việt chỉ search một lần bằng tiếng Việt.** Với q21 ("Độ co giãn của cung theo giá…"), baseline search nguyên câu tiếng Việt trên corpus tiếng Anh nên trượt trang 118.
  - Cách thử: cho phép tối đa 3 lượt search và nhắc trong prompt dịch câu hỏi sang tiếng Anh. Ở cấu hình `agent_hybrid_score` và `agent_hybrid_score_filter`, agent search lần hai bằng "Calculating the Price Elasticity of Supply".
  - Trên test, đây chính là chỗ khác biệt: baseline từ chối t46 (câu tiếng Việt) mà không search lần nào, còn cấu hình chọn dịch query và trả lời đúng.

**Đánh đổi:** cấu hình chọn tốn khoảng 1,7 lần token so với baseline (21.121 so với 12.536 token prompt trên test) và chậm hơn 0,6 s/câu, nhưng tăng CER test từ 6/8 lên 8/8. Bước verify với `gpt-4o-mini` không chặn câu nào ở dev, chỉ thêm 27% chi phí và 60% thời gian, nên không chọn cho phần cơ bản.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Có ba tool: `search(query)`, `answer(answer, citations)` và `abstain(reason)`. Mỗi lượt model phải gọi đúng một tool (`tool_choice=required`).
  - Agent dừng khi model gọi answer hoặc abstain. Khi hết `max_searches`, model buộc phải gọi answer hoặc abstain, và tổng số lượt gọi model bị giới hạn.
  - Context được chọn lại sau mỗi lần search từ toàn bộ ứng viên, theo score, cho đến khi hết budget ký tự. Chỉ tool message mới nhất mang context, các message cũ được thay bằng một dòng ghi chú để token không nhân lên.
  - Citation phải là ID có trong context cuối và phải xuất hiện dạng `[id]` trong câu trả lời. Agent được sửa một lần, sau đó từ chối.
  - Lỗi API hoặc vector store được ghi `status: error` kèm thông báo, không bao giờ thay bằng câu trả lời giả. Lần chạy test không có lỗi nào.
  - Filter chỉ lấy từ trường `filter` của câu hỏi, tức thông tin người dùng cho trước. Gold, đáp án và `absence_reason` không bao giờ được gửi vào model hay retriever.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline / test | 10/11 | 1 search: 10 câu, 0 search: 1 câu | 2 / 1 / 5, trung bình 1,375 | 6/6 | 6/6 | 3/3 | 2/8 |
| Đã chọn / test | 11/11 | 1 search: 10 câu, 2 search: 1 câu | 0 / 1 / 7, trung bình 1,875 | 9/9 | 8/8 | 3/3 | 0/8 |

Grounded answer chấm trên 8 câu test có đáp án, thang 0/1/2. Cấu hình chọn: 7 câu mức 2, 1 câu mức 1 (t47). Baseline: 5 câu mức 2, 1 câu mức 1 (t47) và 2 câu mức 0 (t41 lấy p.94/112/96 thay vì p.111; t46 không search). Nhãn trong `reviews.json` do AI (Opus) soạn nháp theo trace và trang gốc, tôi kiểm tra lại từng câu. Context: cấu hình chọn có đủ evidence ở cả 8 câu, baseline 6/8.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t41 | numeric | Đã chọn: 1 search "median salary for nurses 2014"; chunk p.110–111 vào context. Baseline: dense top-3 ra p.94/112/96 | Đã chọn: "$67,490 [1db13b…]". Baseline từ chối | Đã chọn 2, citation đúng (chunk có câu "In 2014, the median salary for nurses was $67,490"). Baseline 0, retrieval trượt |
| t46 | cross_lingual | Đã chọn dịch thành "summer weather 2015 favorable salmon fishing…" và lấy được p.72. Baseline gọi abstain luôn | Đã chọn nêu đủ 3 lý do (mưa, nước mát, biển lặng) [7a9d…] | Đã chọn 2, nhưng trả lời bằng tiếng Anh dù hỏi tiếng Việt. Baseline 0, không search |
| t42 | numeric | 1 search, p.135 | 810.000 thuê bao hủy, từ 24,6 triệu xuống 23,79 triệu [4f9c…] | 2. Thêm chi tiết giá $10 → $15,98 có trong context (p.117) nhưng không trích |
| t47 | condition, filter chương 4 | 1 search, p.111 | Lương tăng, số y tá không chắc chắn [7b33…] | 1: hai ý chính đúng (p.111), nhưng câu thêm về vị trí lượng cân bằng mới bị lẫn "ban đầu" với "trước khi dịch chuyển". Hạ từ 2 xuống 1 khi review |
| t49 | thiếu evidence | 1 search "U.S. federal minimum wage 2022" | Từ chối | Đúng: sách chỉ có số liệu đến 2016 |
| t51 | thiếu evidence | 2 search | Từ chối | Đúng: khái niệm vĩ mô, không có trong chương 3–5 |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Retrieval: `results/retrieval/<phương pháp>-<split>/` gồm `traces.json` (top-10, score, rank, hit) và `summary.json`.
  - Agent: `results/agent/<cấu hình>-<split>/` gồm `traces.json` (mọi tool call, context, câu trả lời, citation, token, chi phí, thời gian), `summary.json`, `cases.json`, và với test thêm `reviews.json` (nhãn thủ công) và `agent_metrics.json`.
  - Lệnh chạy, từ thư mục gốc, dùng Python có `openai` và `chromadb`, đặt `PYTHONUTF8=1`:
    - `python -m src.day7.export_day6`
    - `python -m src.day7.run_retrieval --method dense|bm25|hybrid --split dev [--use-filter] --name NAME`
    - `python -m src.day7.run_agent_bench --config configs/agent_hybrid_score_filter.json --split test`
    - `python starter/agent_metrics.py --cases … --traces … --reviews … --split test --output …`
    - `python -m unittest discover -s tests -v` (29 test).
  - Toàn bộ thí nghiệm cơ bản tốn dưới $0,15, gồm khoảng 10 lượt chạy dev và 2 lượt test.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - Trên test, cấu hình chọn hơn baseline 2 câu: t41 nhờ hybrid lấy được p.111, t46 nhờ dịch query.
  - Test chỉ có 8 câu có đáp án, nên chênh một câu là 12,5 điểm phần trăm.
  - Tôi đã chạy dev nhiều lần trong khi sửa agent. Sau khi sửa xong, tôi chạy lại toàn bộ 5 cấu hình dev trên cùng một phiên bản code rồi mới chọn.
  - Sách có chỗ tự mâu thuẫn: $67,490 là lương năm 2014 ở p.111 nhưng năm 2015 ở p.93.
  - Filter chỉ có tác dụng ở 7 câu dev và 1 câu test có filter. Filter do người viết câu hỏi (tôi) gán; ở t47 filter trùng chương của gold nhưng không làm đổi kết quả đo được. Phần lớn mức tăng đến từ hybrid và cách chọn context theo score.

## Thử thách nâng cao

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:**
  - OpenStax *Principles of Economics 2e*, bản PDF 974 trang (openstax.org, collection col12122 v1.4), CC BY 4.0. Grader tải được bản gốc miễn phí.
  - Có khoảng 201 bảng và 288 hình có chú thích (đếm theo mẫu chú thích dùng trong `advanced/index_visual.py`), nhiều trong số đó là biểu đồ số liệu.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - Text: tách từng trang bằng PyMuPDF, bỏ header và footer lặp lại, chunk khoảng 1.200 ký tự với overlap 150, không nối qua trang. Được 3.048 chunk.
  - Nội dung trực quan: chọn 645 trang có ảnh raster, có từ 15 nét vẽ vector trở lên, hoặc có dòng "Table/Figure N.N". Mỗi trang render 150 dpi rồi gửi kèm text trang cho `gpt-4.1-mini` vision, yêu cầu JSON:
    - bảng: chép lại đầy đủ dạng markdown;
    - biểu đồ: trục, đơn vị, từng series và giá trị đọc được;
    - sơ đồ: mọi nhãn và mũi tên.
  - Được 793 chunk trực quan: bảng 288, biểu đồ 214, sơ đồ 102, ảnh 71, khác 118. Mỗi chunk giữ trang, `region` (vd "Figure 30.8") và `caption_bbox` để tìm lại hình gốc.
  - Không trang nào lỗi; 12 trang ứng viên không có nội dung hữu ích.
  - Ảnh trang trí, ảnh chụp và mã QR dưới 40 ký tự mô tả thì bỏ.
  - Thống kê đầy đủ ở `advanced/data/index_stats.json`.
  - Chi phí: vision $1,38, embedding $0,016. Embed khoảng 41 s, vision khoảng 5 phút với 8 luồng.
  - Giới hạn chính: biểu đồ đường thường chỉ được chép giá trị ở các mốc thập kỷ, nên có thể mất đỉnh hoặc đáy nằm giữa hai mốc.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - File `advanced/data/questions.json` gồm 49 câu, viết trước khi index:
    - test 30 câu: bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4, thiếu evidence 6;
    - dev 19 câu: bảng, ảnh/sơ đồ, biểu đồ, multi-hop mỗi nhóm 3; text 4 (gồm ad16 sau khi gán nhãn lại); thiếu evidence 3 (thêm ad19 thay ad16).
  - Mỗi gold có trang, `region` và `evidence`. Evidence là text nguyên văn (đã kiểm tra bằng script, không sai câu nào) hoặc mô tả hình đọc từ ảnh trang gốc (AI soạn nháp, tôi kiểm tra). Câu biểu đồ có khoảng chấp nhận ghi trong `grading_note`. Câu multi-hop dùng `gold_rule: all`.
  - Trên dev tôi sửa một lỗi nhãn: ad16 ban đầu ghi là thiếu evidence, nhưng p.399 có ghi "raised from $100,000 to $250,000 in 2008". Phần test không thay đổi (sha256 của phần test sau khi sửa: `847c4036…`, xem LOG T62; bản file trước khi sửa không được lưu nên chỉ ghi lại được kết quả so sánh lúc đó).
  - Cấu hình chốt trên dev, `configs/advanced_gpt41.json`:
    - hybrid, top-6, tối đa 4 search, 9.000 ký tự, chọn context theo score;
    - `expand_region`: tự thêm phần còn lại của cùng bảng hoặc hình ở trang liền kề;
    - model trả lời `gpt-4.1`; bước verify dùng `gpt-4.1`.
  - Tỷ lệ đạt dev theo từng bước:
    - `gpt-4.1-mini` không verify: 16/19.
    - Thêm verify (model chấm `gpt-4.1-mini`): 17/19.
    - Đổi model chấm sang `gpt-4.1`: 18/19; thêm `expand_region`: 18/19.
    - Đổi model trả lời sang `gpt-4.1`: 19/19.
    - Nhãn dev từng câu: `results/advanced/dev_labels.json`.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6/6 | 100% | at05 nằm ở phần Table 6.1 trên p.145, `expand_region` lấy đủ cả hai trang |
| Ảnh, sơ đồ | 5/5 | 100% | |
| Biểu đồ | 3/5 | 60% | at12 đúng mức 27–28% nhưng sai thời điểm đỉnh ("cuối 1940s–đầu 1950s", thực tế khoảng 1954). at16 nói đáy khoảng 30% năm 1980, thực tế khoảng 24% năm 1974. Cả hai do bản chép biểu đồ đường thiếu đỉnh/đáy |
| Multi-hop | 4/4 | 100% | Phép tính hiện rõ và trích đủ hai nguồn; at18 cần cả hai phần Table 6.1 (p.144–145) |
| Text | 4/4 | 100% | |
| Thiếu bằng chứng | 6/6 | 100% | at30 nêu "Đức 2006 = 10,4%" trong lý do từ chối. Số này sách ghi ở p.784 (Table 21.4 lại ghi 9,8%), nên tôi tính là đạt |
| **Toàn bộ** | **28/30** | **93,3%** | 0 lỗi hệ thống; 152 s cho 30 câu (trung bình 5,1 s); $0,30 |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:**
  - Trace: `results/advanced/agent/advanced_gpt41-test/traces.json`.
  - Nhãn và lý do từng câu: `results/advanced/test_labels.json`; AI (Opus) soạn nháp đối chiếu với trang gốc, tôi kiểm tra lại.
  - Retrieval trên test với câu hỏi gốc, 24 câu có đáp án:
    - dense: Hit@1 13, Hit@3 20, Hit@10 24;
    - BM25: 13 / 20 / 23;
    - hybrid: 14 / 21 / 24.
  - Với gold dạng hình, Hit là mức trang: chunk nào trên đúng trang cũng tính là hit. Gold dạng text hoặc bảng vẫn phải nằm trong chunk hoặc khớp chunk hình cùng `region`.
- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - Tái lập: `python -m advanced.index_text`, rồi `python -m advanced.index_visual` (cache từng trang trong `advanced/data/cache/vision/`), rồi `python -m advanced.build_index`. Sau đó chạy `run_agent_bench` với `--questions advanced/data/questions.json --chunks advanced/data/chunks.json --chroma advanced/chroma_db --collection econ2e_full`. Chi tiết trong `advanced/README.md`.
  - Giới hạn:
    - Bản chép vision chỉ được kiểm tra ở các hình có trong benchmark.
    - 28/30 phụ thuộc vào nhãn at30. Nếu tính at30 là không đạt thì còn 27/30 = 90,0%, vẫn đạt ngưỡng.
    - Model chấm là cùng họ với model trả lời.
    - Một số câu test dùng lại bảng hoặc hình đã có trong câu dev, nên các sửa trên dev (`expand_region`, model `gpt-4.1`) có lợi trực tiếp cho at05 và at18.
    - Câu at19 tự cho sẵn giá trị $7,25 trong câu hỏi.
    - Embedding của câu hỏi được cache, nên lần chạy lại tính $0 cho phần này.
