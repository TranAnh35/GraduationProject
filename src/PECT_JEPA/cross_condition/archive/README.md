# Historical material

`historical_audits/` chứa các script local trước đây ở `_audit/`. Chúng được giữ
để truy xuất, không thuộc entrypoint được hỗ trợ và không được tự động chạy bởi
campaign. Các script này có thể phụ thuộc đường dẫn ở vị trí cũ; cần xem lại
trước khi tái sử dụng. Không chuyển checkpoint hay kết quả cũ vào archive.

Các script trong `experiments/`, `colab/`, `verification/` và bộ `tests/` cũ
vẫn được giữ để không phá replay hoặc công việc đang có. Chúng không được nhập
vào luồng nghiên cứu mới. Chỉ `campaign/verification.py` là kiểm tra acceptance
của campaign mới.
