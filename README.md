# CVAE for Trajectory Planning
Build the image of the Docker container:
```bash
docker build --no-cache -t sparse_planner .
```
Run the container:
```bash
docker run -it --name sparse_planner_container --gpus all sparse_planner bash
```
