# Contributing

Thank you for improving this project! Please follow these guidelines.

# Pull Requests

Proposed workflow

```bash
# Work from your local repository
cd boxmot
pip install uv
uv sync --all-extras --all-groups  # installs boxmot in editable mode with all dependencies

# Create a branch
git checkout -b feature/short-desc

# Develop
# ...

# Run functionality where changes were introduced
uv run boxmot track --detector yolov8x --reid osnet_x0_25_msmt17 --tracker bytetrack --source my_video.mp4 --classes 0
uv run boxmot generate --detector yolov8x --reid osnet_x0_25_msmt17 --source path/to/dataset --classes 0
uv run boxmot eval --benchmark mot17 --split ablation --tracker bytetrack
uv run boxmot tune --benchmark mot17 --split ablation --tracker bytetrack

# Run tests
uv run pytest

# Commit & push
git add .
git commit -m "type: summary"
git push origin feature/short-desc

# Share the branch through your team's normal review process
```
