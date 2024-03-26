docker run -it \
       --gpus all \
       -v $(pwd):/workspace \
       example-repo:pytorch2.2.0-cuda12.1  \
       /bin/bash