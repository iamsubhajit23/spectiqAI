# 🔍 SpectiqAI Checkpoint Audit Report

This report presents an audit of the PyTorch model checkpoints stored in the `models/` directory of the SpectiqAI project. This audit was performed to optimize storage usage and prepare the repository for deployment by removing intermediate/redundant checkpoints while preserving production checkpoints.

## Checkpoint Usage Summary

| File Name | File Size | Used By | Safe to Delete? | Reason / Action |
| :--- | :--- | :--- | :--- | :--- |
| `models/simple_cnn/simple_cnn_epoch_1.pt` | 1.15 MB (1,206,789 bytes) | None | **Yes** | Intermediate checkpoint from epoch 1. The application defaults to epoch 3. |
| `models/simple_cnn/simple_cnn_epoch_2.pt` | 1.15 MB (1,206,789 bytes) | None | **Yes** | Intermediate checkpoint from epoch 2. The application defaults to epoch 3. |
| `models/simple_cnn/simple_cnn_epoch_3.pt` | 1.15 MB (1,206,789 bytes) | `app/streamlit_app.py`, `src/spectiqai/config.py`, `config.json` | **No** | **Production Checkpoint**. Default model used for CNN inference and validation. |
| `models/transfer_learning/resnet50/resnet50_epoch_1.pt` | 90.03 MB (94,400,031 bytes) | None | **Yes** | Intermediate checkpoint from epoch 1. The application defaults to epoch 5. |
| `models/transfer_learning/resnet50/resnet50_epoch_2.pt` | 90.03 MB (94,400,031 bytes) | None | **Yes** | Intermediate checkpoint from epoch 2. The application defaults to epoch 5. |
| `models/transfer_learning/resnet50/resnet50_epoch_3.pt` | 90.03 MB (94,400,031 bytes) | None | **Yes** | Intermediate checkpoint from epoch 3. The application defaults to epoch 5. |
| `models/transfer_learning/resnet50/resnet50_epoch_4.pt` | 90.03 MB (94,400,031 bytes) | None | **Yes** | Intermediate checkpoint from epoch 4. The application defaults to epoch 5. |
| `models/transfer_learning/resnet50/resnet50_epoch_5.pt` | 90.03 MB (94,400,031 bytes) | `app/streamlit_app.py` (model comparison tab) | **No** | **Production Checkpoint**. Required for ResNet50 evaluations and comparisons in the UI. |
| `models/transfer_learning/efficientnet_b0/efficientnet_b0_epoch_1.pt` | 15.60 MB (16,362,723 bytes) | None | **Yes** | Intermediate checkpoint from epoch 1. The application defaults to epoch 5. |
| `models/transfer_learning/efficientnet_b0/efficientnet_b0_epoch_2.pt` | 15.60 MB (16,362,723 bytes) | None | **Yes** | Intermediate checkpoint from epoch 2. The application defaults to epoch 5. |
| `models/transfer_learning/efficientnet_b0/efficientnet_b0_epoch_3.pt` | 15.60 MB (16,362,723 bytes) | None | **Yes** | Intermediate checkpoint from epoch 3. The application defaults to epoch 5. |
| `models/transfer_learning/efficientnet_b0/efficientnet_b0_epoch_4.pt` | 15.60 MB (16,362,723 bytes) | None | **Yes** | Intermediate checkpoint from epoch 4. The application defaults to epoch 5. |
| `models/transfer_learning/efficientnet_b0/efficientnet_b0_epoch_5.pt` | 15.60 MB (16,362,723 bytes) | `app/streamlit_app.py` (model comparison tab) | **No** | **Production Checkpoint**. Required for EfficientNet evaluations and comparisons in the UI. |

## Storage Reclamation Impact

- **Total Initial Model Directory Size:** 531.60 MB (13 files)
- **Total Reclaimed Space:** 424.82 MB (10 files deleted)
- **Total Production Models Retained:** 106.78 MB (3 files retained)
- **Reduction Rate:** **79.91%**

*Report generated on 2026-06-22.*
