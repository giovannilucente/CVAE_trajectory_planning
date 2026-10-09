# CVAE for Trajectory Planning
Build the image of the Docker container:
```bash
docker build --no-cache -t sparse_planner .
```
Create and start the container:
```bash
docker run -it --name sparse_planner_container --gpus all sparse_planner bash
```
To train the model:
```bash
python3 train.py
```
